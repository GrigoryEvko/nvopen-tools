// Function: sub_22595D0
// Address: 0x22595d0
//
__int64 *__fastcall sub_22595D0(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v4; // rbx
  __int64 *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // r14
  _QWORD *v10; // r13
  _QWORD *v11; // rcx
  size_t v13; // r14
  size_t v14; // r15
  size_t v15; // rdx
  int v16; // eax
  unsigned int v17; // edi
  __int64 v18; // r14
  unsigned __int64 v19; // rdi

  v4 = a1;
  v5 = (__int64 *)sub_22077B0(0x48u);
  v6 = *a3;
  v5[4] = (__int64)(v5 + 6);
  sub_2257AB0(v5 + 4, *(_BYTE **)v6, *(_QWORD *)v6 + *(_QWORD *)(v6 + 8));
  v5[8] = 0;
  v7 = sub_2259340(a1, a2, (__int64)(v5 + 4));
  v9 = v7;
  if ( v8 )
  {
    v10 = v8;
    v11 = a1 + 1;
    LOBYTE(a1) = 1;
    if ( v7 || v8 == v11 )
      goto LABEL_3;
    v13 = v5[5];
    v15 = v8[5];
    v14 = v15;
    if ( v13 <= v15 )
      v15 = v5[5];
    if ( !v15 || (v16 = memcmp((const void *)v5[4], (const void *)v10[4], v15), v11 = v4 + 1, (v17 = v16) == 0) )
    {
      v18 = v13 - v14;
      LOBYTE(a1) = 0;
      if ( v18 > 0x7FFFFFFF )
      {
LABEL_3:
        sub_220F040((char)a1, (__int64)v5, v10, v11);
        ++v4[5];
        return v5;
      }
      if ( v18 < (__int64)0xFFFFFFFF80000000LL )
      {
        LOBYTE(a1) = 1;
        goto LABEL_3;
      }
      v17 = v18;
    }
    LODWORD(a1) = v17 >> 31;
    goto LABEL_3;
  }
  v19 = v5[4];
  if ( v5 + 6 != (__int64 *)v19 )
    j_j___libc_free_0(v19);
  j_j___libc_free_0((unsigned __int64)v5);
  return (__int64 *)v9;
}
