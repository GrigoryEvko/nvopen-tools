// Function: sub_2735F90
// Address: 0x2735f90
//
void __fastcall sub_2735F90(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r13
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 *v14; // rbx
  unsigned __int64 v15; // r14
  __int64 *v16; // r14
  __int64 *v17; // rbx
  __int64 *v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  char *v24; // r8
  char v25; // cl
  __int64 *v26; // r8
  __int64 v27; // rcx
  __int64 *v28; // [rsp+8h] [rbp-78h]
  unsigned __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 *v31; // [rsp+20h] [rbp-60h]
  __int64 v32; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34; // [rsp+38h] [rbp-48h]
  unsigned __int64 *v35; // [rsp+40h] [rbp-40h]

  v32 = a2;
  if ( a2 )
  {
    v31 = (__int64 *)sub_2735BC0((__int64)(a1 + 11), &v32, a3);
    if ( v32 )
    {
      v8 = sub_2735710((__int64)(a1 + 691), &v32, v4, v5, v6, v7);
      goto LABEL_4;
    }
  }
  else
  {
    v31 = a1 + 8;
  }
  v8 = (__int64)(a1 + 17);
LABEL_4:
  v9 = v31[1];
  v10 = *v31;
  sub_2732E30((__int64 *)&v33, *v31, 0xCF3CF3CF3CF3CF3DLL * ((v9 - *v31) >> 3));
  if ( v35 )
    sub_2730090(v10, v9, (__int64)v35, v34, v12);
  else
    sub_272E800(v10, v9, 0, v11, v12, v13);
  v14 = v35;
  v15 = (unsigned __int64)&v35[21 * v34];
  if ( v35 != (unsigned __int64 *)v15 )
  {
    do
    {
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        _libc_free(*v14);
      v14 += 21;
    }
    while ( (unsigned __int64 *)v15 != v14 );
    v15 = (unsigned __int64)v35;
  }
  j_j___libc_free_0(v15);
  v16 = (__int64 *)*v31;
  v17 = (__int64 *)v31[1];
  v18 = (__int64 *)(*v31 + 168);
  if ( v17 != v18 )
  {
    v19 = *v31 + 168;
    while ( 1 )
    {
      v20 = *(_QWORD *)(v19 + 144);
      v21 = v16[18];
      if ( *(_QWORD *)(v21 + 8) != *(_QWORD *)(v20 + 8) )
        goto LABEL_13;
      v22 = *(_QWORD *)v19;
      v23 = *(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8);
      if ( *(_QWORD *)v19 == v23 )
      {
LABEL_27:
        v26 = (__int64 *)(v21 + 24);
        v30 = 0;
        LODWORD(v34) = *(_DWORD *)(v20 + 32);
        if ( (unsigned int)v34 <= 0x40 )
          goto LABEL_20;
      }
      else
      {
        while ( 1 )
        {
          v24 = *(char **)v22;
          v25 = **(_BYTE **)v22;
          if ( v25 == 61 )
          {
            v30 = *((_QWORD *)v24 + 1);
            goto LABEL_19;
          }
          if ( v25 == 62 && *(_QWORD *)&v24[32 * *(unsigned int *)(v22 + 8) - 64] == *((_QWORD *)v24 - 4) )
            break;
          v22 += 16;
          if ( v23 == v22 )
            goto LABEL_27;
        }
        v30 = *(_QWORD *)(*((_QWORD *)v24 - 8) + 8LL);
LABEL_19:
        v26 = (__int64 *)(v21 + 24);
        LODWORD(v34) = *(_DWORD *)(v20 + 32);
        if ( (unsigned int)v34 <= 0x40 )
        {
LABEL_20:
          v33 = *(_QWORD *)(v20 + 24);
          goto LABEL_21;
        }
      }
      v28 = v26;
      sub_C43780((__int64)&v33, (const void **)(v20 + 24));
      v26 = v28;
LABEL_21:
      sub_C46B40((__int64)&v33, v26);
      if ( (unsigned int)v34 <= 0x40 )
      {
        v27 = 0;
        if ( (_DWORD)v34 )
          v27 = (__int64)(v33 << (64 - (unsigned __int8)v34)) >> (64 - (unsigned __int8)v34);
        v29 = v27;
        if ( (unsigned __int8)sub_DFA0C0(*a1) && (!v30 || sub_DFA150((__int64 *)*a1, v30, 0, v29, 1u, 0)) )
          goto LABEL_14;
      }
      else if ( v33 )
      {
        j_j___libc_free_0_0(v33);
      }
LABEL_13:
      sub_2731C00((__int64)a1, v16, (__int64 *)v19, v8);
      v16 = (__int64 *)v19;
LABEL_14:
      v19 += 168;
      if ( v17 == (__int64 *)v19 )
      {
        v18 = (__int64 *)v31[1];
        break;
      }
    }
  }
  sub_2731C00((__int64)a1, v16, v18, v8);
}
