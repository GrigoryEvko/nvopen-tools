// Function: sub_8EE880
// Address: 0x8ee880
//
__int64 __fastcall sub_8EE880(_BYTE *a1, int a2, int a3)
{
  int v5; // esi
  __int64 result; // rax
  int v7; // esi
  int v8; // r8d
  int v9; // r14d
  int v10; // edx
  unsigned __int8 *v11; // rax
  char v12; // r15
  __int64 v13; // r13
  __int64 v14; // r9
  int v15; // edx
  unsigned __int8 *v16; // rdi
  unsigned __int8 *v17; // rdi
  int v18; // edx
  int v19; // edx

  v5 = a2 + 14;
  result = (unsigned int)(a2 + 7);
  if ( a2 + 7 >= 0 )
    v5 = a2 + 7;
  v7 = v5 >> 3;
  if ( a2 > a3 )
  {
    if ( a3 )
    {
      v8 = a3 % 8;
      v9 = v7 - a3 / 8;
      v10 = v7 - 1;
      v11 = &a1[v9 - 1];
      v12 = *v11 << v8;
      a1[v7 - 1] = v12;
      if ( v9 == 1 )
      {
        v13 = v7;
      }
      else
      {
        v13 = v7;
        v14 = (__int64)&a1[v9 - 2 - (v9 - 2)];
        do
        {
          v15 = *(v11 - 1);
          v16 = v11--;
          v17 = &v16[-v9];
          v17[v7] = v12 | (v15 >> (8 - v8));
          v18 = *v11 << v8;
          v17[v7 - 1] = v18;
          v12 = v18;
        }
        while ( (unsigned __int8 *)v14 != v11 );
        v10 = v7 - v9;
      }
      if ( v10 )
        memset(&a1[v10 - 1LL - (unsigned int)(v10 - 1)], 0, (unsigned int)(v10 - 1) + 1LL);
      result = (__int64)&a1[v13 - 1];
      v19 = *(unsigned __int8 *)result;
      if ( (a2 & 7) != 0 )
        v19 &= ~(-1 << (a2 % 8));
      *(_BYTE *)result = v19;
    }
  }
  else if ( a2 > 0 )
  {
    return (__int64)memset(a1, 0, (unsigned int)(v7 - 1) + 1LL);
  }
  return result;
}
