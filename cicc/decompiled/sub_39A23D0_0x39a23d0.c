// Function: sub_39A23D0
// Address: 0x39a23d0
//
unsigned __int8 *__fastcall sub_39A23D0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rcx
  unsigned __int8 *result; // rax
  int v5; // edx
  __int64 v6; // rdi
  int v7; // ecx
  unsigned int v8; // edx
  unsigned __int8 **v9; // rax
  unsigned __int8 *v10; // rsi
  int v11; // eax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  unsigned int v15; // edx
  unsigned __int8 *v16; // rsi
  int v17; // r8d
  int v18; // eax
  int v19; // r8d

  if ( (unsigned __int8)sub_39A2350((_QWORD *)a1, a2) )
  {
    v3 = *(_QWORD *)(a1 + 208);
    result = 0;
    v5 = *(_DWORD *)(v3 + 384);
    if ( !v5 )
      return result;
    v6 = *(_QWORD *)(v3 + 368);
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (unsigned __int8 **)(v6 + 16LL * v8);
    v10 = *v9;
    if ( a2 != *v9 )
    {
      v11 = 1;
      while ( v10 != (unsigned __int8 *)-8LL )
      {
        v17 = v11 + 1;
        v8 = v7 & (v11 + v8);
        v9 = (unsigned __int8 **)(v6 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          return v9[1];
        v11 = v17;
      }
      return 0;
    }
    return v9[1];
  }
  v12 = *(_DWORD *)(a1 + 248);
  result = 0;
  if ( v12 )
  {
    v13 = v12 - 1;
    v14 = *(_QWORD *)(a1 + 232);
    v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (unsigned __int8 **)(v14 + 16LL * v15);
    v16 = *v9;
    if ( a2 != *v9 )
    {
      v18 = 1;
      while ( v16 != (unsigned __int8 *)-8LL )
      {
        v19 = v18 + 1;
        v15 = v13 & (v18 + v15);
        v9 = (unsigned __int8 **)(v14 + 16LL * v15);
        v16 = *v9;
        if ( a2 == *v9 )
          return v9[1];
        v18 = v19;
      }
      return 0;
    }
    return v9[1];
  }
  return result;
}
