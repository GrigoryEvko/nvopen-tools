// Function: sub_A288A0
// Address: 0xa288a0
//
__int64 __fastcall sub_A288A0(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  size_t v4; // r14
  size_t v5; // r15
  const void *v6; // r8
  const void *v7; // r9
  size_t v8; // rdx
  signed __int64 v9; // rax
  _QWORD *v10; // r9
  __int64 v11; // r15
  int v13; // eax
  int v14; // eax
  const void *v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+10h] [rbp-40h]
  const void *v17; // [rsp+18h] [rbp-38h]
  const void *v18; // [rsp+18h] [rbp-38h]
  const void *v19; // [rsp+18h] [rbp-38h]

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_A28730((__int64)a1, a3);
    v14 = sub_A15B80(*(const void **)(a1[4] + 32LL), *(_QWORD *)(a1[4] + 40LL), *(const void **)a3, *(_QWORD *)(a3 + 8));
    v10 = 0;
    if ( v14 >= 0 )
      return sub_A28730((__int64)a1, a3);
    return (__int64)v10;
  }
  v4 = *(_QWORD *)(a3 + 8);
  v5 = a2[5];
  v6 = *(const void **)a3;
  v7 = (const void *)a2[4];
  v8 = v5;
  if ( v4 <= v5 )
    v8 = v4;
  if ( !v8 || (v15 = (const void *)a2[4], v17 = v6, LODWORD(v9) = memcmp(v6, v15, v8), v6 = v17, v7 = v15, !(_DWORD)v9) )
  {
    v9 = v4 - v5;
    if ( (__int64)(v4 - v5) > 0x7FFFFFFF )
      goto LABEL_12;
    if ( v9 < (__int64)0xFFFFFFFF80000000LL )
      goto LABEL_9;
  }
  if ( (int)v9 < 0 )
  {
LABEL_9:
    v10 = a2;
    if ( (_QWORD *)a1[3] != a2 )
    {
      v18 = v6;
      v11 = sub_220EF80(a2);
      if ( sub_A15B80(*(const void **)(v11 + 32), *(_QWORD *)(v11 + 40), v18, v4) >= 0 )
        return sub_A28730((__int64)a1, a3);
      v10 = 0;
      if ( *(_QWORD *)(v11 + 24) )
        return (__int64)a2;
    }
    return (__int64)v10;
  }
LABEL_12:
  v19 = v6;
  v13 = sub_A15B80(v7, v5, v6, v4);
  v10 = a2;
  if ( v13 < 0 )
  {
    if ( (_QWORD *)a1[4] == a2 )
    {
      return 0;
    }
    else
    {
      v16 = sub_220EEE0(a2);
      if ( sub_A15B80(v19, v4, *(const void **)(v16 + 32), *(_QWORD *)(v16 + 40)) >= 0 )
        return sub_A28730((__int64)a1, a3);
      v10 = 0;
      if ( a2[3] )
        return v16;
    }
  }
  return (__int64)v10;
}
