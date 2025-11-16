// Function: sub_12BFA60
// Address: 0x12bfa60
//
__int64 *__fastcall sub_12BFA60(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  __int64 *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // r14
  _QWORD *v10; // r13
  _QWORD *v11; // rcx
  __int64 v12; // rdi
  size_t v14; // r14
  size_t v15; // r15
  size_t v16; // rdx
  int v17; // eax
  unsigned int v18; // edi
  __int64 v19; // r14
  __int64 *v20; // rdi

  v5 = (__int64 *)sub_22077B0(72);
  v6 = *a3;
  v5[4] = (__int64)(v5 + 6);
  sub_12BCB70(v5 + 4, *(_BYTE **)v6, *(_QWORD *)v6 + *(_QWORD *)(v6 + 8));
  v5[8] = 0;
  v7 = sub_12BF7D0(a1, a2, (__int64)(v5 + 4));
  v9 = v7;
  if ( v8 )
  {
    v10 = v8;
    v11 = a1 + 1;
    v12 = 1;
    if ( v7 || v8 == v11 )
      goto LABEL_3;
    v14 = v5[5];
    v16 = v8[5];
    v15 = v16;
    if ( v14 <= v16 )
      v16 = v5[5];
    if ( !v16 || (v17 = memcmp((const void *)v5[4], (const void *)v10[4], v16), v11 = a1 + 1, (v18 = v17) == 0) )
    {
      v19 = v14 - v15;
      v12 = 0;
      if ( v19 > 0x7FFFFFFF )
      {
LABEL_3:
        sub_220F040(v12, v5, v10, v11);
        ++a1[5];
        return v5;
      }
      if ( v19 < (__int64)0xFFFFFFFF80000000LL )
      {
        v12 = 1;
        goto LABEL_3;
      }
      v18 = v19;
    }
    v12 = v18 >> 31;
    goto LABEL_3;
  }
  v20 = (__int64 *)v5[4];
  if ( v5 + 6 != v20 )
    j_j___libc_free_0(v20, v5[6] + 1);
  j_j___libc_free_0(v5, 72);
  return (__int64 *)v9;
}
