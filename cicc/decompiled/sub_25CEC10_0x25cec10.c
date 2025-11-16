// Function: sub_25CEC10
// Address: 0x25cec10
//
__int64 __fastcall sub_25CEC10(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  size_t v4; // r14
  size_t v5; // r15
  void *v6; // r8
  const void *v7; // r9
  size_t v8; // rdx
  int v9; // eax
  signed __int64 v10; // rax
  signed __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // rax
  size_t v14; // rdx
  size_t v15; // r15
  _QWORD *v16; // rcx
  int v17; // eax
  __int64 v18; // r15
  __int64 v19; // r15
  size_t n; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+10h] [rbp-40h]
  _QWORD *s1a; // [rsp+10h] [rbp-40h]
  void *s2a; // [rsp+18h] [rbp-38h]
  void *s2b; // [rsp+18h] [rbp-38h]
  void *s2; // [rsp+18h] [rbp-38h]
  void *s2c; // [rsp+18h] [rbp-38h]

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5]
      || sub_25CD000(*(const void **)(a1[4] + 32LL), *(_QWORD *)(a1[4] + 40LL), *(const void **)a3, *(_QWORD *)(a3 + 8)) >= 0 )
    {
      return sub_25CEAA0((__int64)a1, a3);
    }
    return 0;
  }
  v4 = *(_QWORD *)(a3 + 8);
  v5 = a2[5];
  v6 = *(void **)a3;
  v7 = (const void *)a2[4];
  v8 = v5;
  if ( v4 <= v5 )
    v8 = v4;
  if ( !v8 )
  {
    v10 = v4 - v5;
    if ( (__int64)(v4 - v5) > 0x7FFFFFFF )
      goto LABEL_11;
LABEL_7:
    if ( v10 >= (__int64)0xFFFFFFFF80000000LL && (int)v10 >= 0 )
    {
      if ( !v8 )
        goto LABEL_11;
      goto LABEL_10;
    }
LABEL_19:
    result = (__int64)a2;
    if ( (_QWORD *)a1[3] == a2 )
      return result;
    s2 = v6;
    v13 = sub_220EF80((__int64)a2);
    v14 = v4;
    v15 = *(_QWORD *)(v13 + 40);
    v16 = (_QWORD *)v13;
    if ( v15 <= v4 )
      v14 = *(_QWORD *)(v13 + 40);
    if ( !v14 || (s1a = (_QWORD *)v13, v17 = memcmp(*(const void **)(v13 + 32), s2, v14), v16 = s1a, !v17) )
    {
      v18 = v15 - v4;
      if ( v18 > 0x7FFFFFFF )
        return sub_25CEAA0((__int64)a1, a3);
      if ( v18 < (__int64)0xFFFFFFFF80000000LL )
        goto LABEL_28;
      v17 = v18;
    }
    if ( v17 < 0 )
    {
LABEL_28:
      result = 0;
      if ( v16[3] )
        return (__int64)a2;
      return result;
    }
    return sub_25CEAA0((__int64)a1, a3);
  }
  n = v8;
  s1 = (void *)a2[4];
  s2a = v6;
  v9 = memcmp(v6, s1, v8);
  v6 = s2a;
  v7 = s1;
  v8 = n;
  if ( v9 )
  {
    if ( v9 < 0 )
      goto LABEL_19;
  }
  else
  {
    v10 = v4 - v5;
    if ( (__int64)(v4 - v5) <= 0x7FFFFFFF )
      goto LABEL_7;
  }
LABEL_10:
  s2b = v6;
  LODWORD(v11) = memcmp(v7, v6, v8);
  v6 = s2b;
  if ( (_DWORD)v11 )
  {
LABEL_13:
    if ( (int)v11 >= 0 )
      return (__int64)a2;
    goto LABEL_34;
  }
LABEL_11:
  v11 = v5 - v4;
  if ( (__int64)(v5 - v4) > 0x7FFFFFFF )
    return (__int64)a2;
  if ( v11 >= (__int64)0xFFFFFFFF80000000LL )
    goto LABEL_13;
LABEL_34:
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  s2c = v6;
  v19 = sub_220EEE0((__int64)a2);
  if ( sub_25CD000(s2c, v4, *(const void **)(v19 + 32), *(_QWORD *)(v19 + 40)) >= 0 )
    return sub_25CEAA0((__int64)a1, a3);
  result = 0;
  if ( a2[3] )
    return v19;
  return result;
}
