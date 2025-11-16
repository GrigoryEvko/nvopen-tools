// Function: sub_153CF40
// Address: 0x153cf40
//
__int64 __fastcall sub_153CF40(__int64 **a1, __int64 **a2, __int64 **a3, __int64 **a4, __int64 a5, __int64 a6)
{
  __int64 **v7; // r12
  __int64 **v8; // rbx
  __int64 *v9; // rax
  __int64 *v10; // rax
  signed __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rcx
  __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a3;
  v8 = a1;
  v21[0] = a6;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      if ( sub_153CA80((__int64)v21, v7, v8) )
      {
        v9 = *v7;
        a5 += 16;
        v7 += 2;
        *(_QWORD *)(a5 - 16) = v9;
        *(_DWORD *)(a5 - 8) = *((_DWORD *)v7 - 2);
        if ( v8 == a2 )
          break;
      }
      else
      {
        v10 = *v8;
        v8 += 2;
        a5 += 16;
        *(_QWORD *)(a5 - 16) = v10;
        *(_DWORD *)(a5 - 8) = *((_DWORD *)v8 - 2);
        if ( v8 == a2 )
          break;
      }
    }
    while ( v7 != a4 );
  }
  v11 = (char *)a2 - (char *)v8;
  v12 = ((char *)a2 - (char *)v8) >> 4;
  if ( (char *)a2 - (char *)v8 > 0 )
  {
    v13 = a5;
    do
    {
      v14 = *v8;
      v13 += 16;
      v8 += 2;
      *(_QWORD *)(v13 - 16) = v14;
      *(_DWORD *)(v13 - 8) = *((_DWORD *)v8 - 2);
      --v12;
    }
    while ( v12 );
    a5 += v11;
  }
  v15 = (char *)a4 - (char *)v7;
  v16 = ((char *)a4 - (char *)v7) >> 4;
  if ( (char *)a4 - (char *)v7 > 0 )
  {
    v17 = a5;
    do
    {
      v18 = *v7;
      v17 += 16;
      v7 += 2;
      *(_QWORD *)(v17 - 16) = v18;
      *(_DWORD *)(v17 - 8) = *((_DWORD *)v7 - 2);
      --v16;
    }
    while ( v16 );
    if ( v15 <= 0 )
      v15 = 16;
    a5 += v15;
  }
  return a5;
}
