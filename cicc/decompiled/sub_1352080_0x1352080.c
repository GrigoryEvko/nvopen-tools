// Function: sub_1352080
// Address: 0x1352080
//
_BYTE *__fastcall sub_1352080(unsigned __int8 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rsi
  _QWORD *v4; // rdi
  size_t v5; // rcx
  size_t v6; // r8
  size_t v7; // rdx
  int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // rax
  _WORD *v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  _WORD *v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rax
  _WORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdi
  _BYTE *result; // rax
  size_t v22; // [rsp+8h] [rbp-E8h]
  size_t v23; // [rsp+10h] [rbp-E0h]
  void *s2; // [rsp+20h] [rbp-D0h] BYREF
  size_t v26; // [rsp+28h] [rbp-C8h]
  _QWORD v27[2]; // [rsp+30h] [rbp-C0h] BYREF
  void *s1; // [rsp+40h] [rbp-B0h] BYREF
  size_t n; // [rsp+48h] [rbp-A8h]
  _QWORD v30[2]; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD v31[4]; // [rsp+60h] [rbp-90h] BYREF
  int v32; // [rsp+80h] [rbp-70h]
  void **p_s2; // [rsp+88h] [rbp-68h]
  _QWORD v34[4]; // [rsp+90h] [rbp-60h] BYREF
  int v35; // [rsp+B0h] [rbp-40h]
  void **p_s1; // [rsp+B8h] [rbp-38h]

  p_s2 = &s2;
  v31[0] = &unk_49EFBE0;
  v34[0] = &unk_49EFBE0;
  s2 = v27;
  v26 = 0;
  LOBYTE(v27[0]) = 0;
  s1 = v30;
  n = 0;
  LOBYTE(v30[0]) = 0;
  v32 = 1;
  memset(&v31[1], 0, 24);
  v35 = 1;
  memset(&v34[1], 0, 24);
  p_s1 = &s1;
  sub_15537D0(a2, v31, 1);
  v3 = v34;
  sub_15537D0(a3, v34, 1);
  sub_16E7BC0(v34);
  v4 = v31;
  sub_16E7BC0(v31);
  v5 = n;
  v6 = v26;
  v7 = v26;
  if ( n <= v26 )
    v7 = n;
  if ( v7 )
  {
    v3 = s2;
    v4 = s1;
    v22 = v26;
    v23 = n;
    v8 = memcmp(s1, s2, v7);
    v5 = v23;
    v6 = v22;
    if ( v8 )
    {
LABEL_8:
      if ( v8 >= 0 )
        goto LABEL_9;
      goto LABEL_26;
    }
  }
  v9 = v5 - v6;
  if ( v9 > 0x7FFFFFFF )
    goto LABEL_9;
  if ( v9 >= (__int64)0xFFFFFFFF80000000LL )
  {
    v8 = v9;
    goto LABEL_8;
  }
LABEL_26:
  v3 = &s1;
  v4 = &s2;
  sub_22415E0(&s2, &s1);
LABEL_9:
  v10 = sub_16E8CB0(v4, v3, v7);
  v11 = *(_WORD **)(v10 + 24);
  v12 = v10;
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v11 <= 1u )
  {
    v12 = sub_16E7EE0(v10, "  ", 2);
  }
  else
  {
    *v11 = 8224;
    *(_QWORD *)(v10 + 24) += 2LL;
  }
  v13 = sub_134CED0(v12, a1);
  v14 = *(_WORD **)(v13 + 24);
  v15 = v13;
  if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 1u )
  {
    v15 = sub_16E7EE0(v13, ":\t", 2);
  }
  else
  {
    *v14 = 2362;
    *(_QWORD *)(v13 + 24) += 2LL;
  }
  v16 = sub_16E7EE0(v15, (const char *)s2, v26);
  v17 = *(_WORD **)(v16 + 24);
  v18 = v16;
  if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 1u )
  {
    v18 = sub_16E7EE0(v16, ", ", 2);
  }
  else
  {
    *v17 = 8236;
    *(_QWORD *)(v16 + 24) += 2LL;
  }
  v19 = sub_16E7EE0(v18, (const char *)s1, n);
  result = *(_BYTE **)(v19 + 24);
  if ( *(_BYTE **)(v19 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v19, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v19 + 24);
  }
  if ( s1 != v30 )
    result = (_BYTE *)j_j___libc_free_0(s1, v30[0] + 1LL);
  if ( s2 != v27 )
    return (_BYTE *)j_j___libc_free_0(s2, v27[0] + 1LL);
  return result;
}
