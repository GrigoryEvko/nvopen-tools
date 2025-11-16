// Function: sub_2042830
// Address: 0x2042830
//
__int64 __fastcall sub_2042830(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  bool (__fastcall *v6)(__int64); // rax
  __int64 *v7; // r12
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r9d
  __int64 *v13; // r14
  int v14; // r13d
  __int64 *v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 *v20; // r13

  v6 = *(bool (__fastcall **)(__int64))(*a1 + 64LL);
  if ( v6 == sub_2041B10 )
  {
    v7 = (__int64 *)a1[6];
    if ( (__int64 *)a1[7] != v7 )
      goto LABEL_3;
    return 0;
  }
  if ( ((unsigned __int8 (*)(void))v6)() )
    return 0;
  v7 = (__int64 *)a1[6];
LABEL_3:
  if ( byte_4FCEDA0 )
  {
    v20 = (__int64 *)a1[7];
    v15 = v7 + 1;
    if ( v20 == v7 + 1 )
      goto LABEL_10;
    do
    {
      if ( sub_2041FF0((__int64)(a1 + 15), *v7, *v15, a4, a5, a6) )
        v7 = v15;
      ++v15;
    }
    while ( v20 != v15 );
  }
  else
  {
    v8 = sub_2042650((__int64)a1, *v7, a3, a4, a5, a6);
    v13 = (__int64 *)a1[7];
    v14 = v8;
    v15 = (__int64 *)(a1[6] + 8LL);
    if ( v13 == v15 )
      goto LABEL_10;
    do
    {
      while ( (int)sub_2042650((__int64)a1, *v15, v9, v10, v11, v12) <= v14 )
      {
        if ( v13 == ++v15 )
          goto LABEL_9;
      }
      v16 = *v15;
      v7 = v15++;
      v14 = sub_2042650((__int64)a1, v16, v9, v10, v11, v12);
    }
    while ( v13 != v15 );
  }
LABEL_9:
  v15 = (__int64 *)a1[7];
LABEL_10:
  v17 = (__int64)(v15 - 1);
  v18 = *v7;
  if ( v7 != v15 - 1 )
  {
    *v7 = *(v15 - 1);
    *(v15 - 1) = v18;
    v17 = a1[7] - 8LL;
  }
  a1[7] = v17;
  return v18;
}
