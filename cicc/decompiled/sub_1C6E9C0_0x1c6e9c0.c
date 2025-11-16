// Function: sub_1C6E9C0
// Address: 0x1c6e9c0
//
__int64 __fastcall sub_1C6E9C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _BOOL4 v4; // r15d
  size_t v6; // rcx
  __int64 v8; // r12
  __int64 v10; // r12
  size_t v11; // rdx
  const char *v12; // rax
  size_t v13; // rdx
  size_t v14; // r8
  size_t v15; // r12
  int v16; // eax
  unsigned int v17; // r15d
  int v18; // eax
  const char *s2; // [rsp+10h] [rbp-40h]
  size_t na; // [rsp+18h] [rbp-38h]
  size_t n; // [rsp+18h] [rbp-38h]

  v4 = 1;
  v6 = a1 + 8;
  if ( a2 || a3 == v6 )
    goto LABEL_2;
  v10 = *a4;
  s2 = sub_1649960(*(_QWORD *)(a3 + 32));
  n = v11;
  v12 = sub_1649960(v10);
  v14 = n;
  v6 = a1 + 8;
  v15 = v13;
  if ( n < v13 )
  {
    v4 = 0;
    if ( !n )
      goto LABEL_2;
    v18 = memcmp(v12, s2, n);
    v14 = n;
    v6 = a1 + 8;
    v17 = v18;
    if ( !v18 )
      goto LABEL_8;
LABEL_11:
    v4 = v17 >> 31;
    goto LABEL_2;
  }
  if ( v13 )
  {
    v16 = memcmp(v12, s2, v13);
    v6 = a1 + 8;
    v14 = n;
    v17 = v16;
    if ( v16 )
      goto LABEL_11;
  }
  v4 = 0;
  if ( v14 != v15 )
LABEL_8:
    v4 = v14 > v15;
LABEL_2:
  na = v6;
  v8 = sub_22077B0(40);
  *(_QWORD *)(v8 + 32) = *a4;
  sub_220F040(v4, v8, a3, na);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
