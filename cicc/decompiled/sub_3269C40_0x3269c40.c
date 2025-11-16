// Function: sub_3269C40
// Address: 0x3269c40
//
__int64 __fastcall sub_3269C40(__int64 **a1, unsigned int a2)
{
  __int64 v4; // rdx
  _QWORD *v5; // rax
  __int64 v6; // rax
  int v7; // edx
  int v8; // edi
  __int64 v9; // rdx
  __int64 *v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 *v14; // rax
  int v15; // edx
  unsigned int v16; // r13d
  __int64 v18; // r12
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  unsigned __int16 *v22; // rdx
  int v23; // eax
  const __m128i *v24; // r13
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 *v29; // r15
  __int64 v30; // rdx
  __m128i *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // r13
  __int64 v34; // rbx
  const __m128i *v35; // rdx
  __int64 v36; // [rsp+8h] [rbp-98h]
  const __m128i *v37[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 (__fastcall *v38)(unsigned __int64 *, const __m128i **, int); // [rsp+40h] [rbp-60h]
  __int64 (__fastcall *v39)(__int64 *, __int64); // [rsp+48h] [rbp-58h]
  unsigned __int64 v40; // [rsp+50h] [rbp-50h] BYREF
  const __m128i *v41; // [rsp+58h] [rbp-48h]
  __int64 (__fastcall *v42)(unsigned __int64 *, const __m128i **, int); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v43)(__int64 *, __int64); // [rsp+68h] [rbp-38h]

  v4 = **a1;
  if ( a2 == 1 && *(_DWORD *)(v4 + 24) == 99 )
    return 0;
  v5 = (_QWORD *)(*(_QWORD *)(v4 + 40) + 40LL * a2);
  v6 = sub_33CF5B0(*v5, v5[1]);
  v8 = v7;
  v9 = v6;
  v10 = a1[1];
  *v10 = v9;
  *((_DWORD *)v10 + 2) = v8;
  v11 = a1[2];
  v12 = *(_QWORD *)(**a1 + 40);
  v13 = v12 + 40LL * (1 - a2);
  *v11 = *(_QWORD *)v13;
  *((_DWORD *)v11 + 2) = *(_DWORD *)(v13 + 8);
  v14 = a1[2];
  v15 = *(_DWORD *)(*v14 + 24);
  if ( v15 == 221 )
    goto LABEL_12;
  v16 = 0;
  if ( v15 != 220 )
    return v16;
  v12 = *a1[3];
  sub_33DD090(&v40, v12, *v14, v14[1], 0);
  v18 = 1LL << ((unsigned __int8)v41 - 1);
  if ( (unsigned int)v41 <= 0x40 )
  {
    v19 = v40 & v18;
    if ( (unsigned int)v43 <= 0x40 )
      goto LABEL_10;
    v20 = (unsigned __int64)v42;
    if ( !v42 )
      goto LABEL_10;
    goto LABEL_9;
  }
  v19 = *(_QWORD *)(v40 + 8LL * ((unsigned int)((_DWORD)v41 - 1) >> 6)) & v18;
  if ( (unsigned int)v43 > 0x40 )
  {
    v20 = (unsigned __int64)v42;
    if ( v42 )
    {
LABEL_9:
      j_j___libc_free_0_0(v20);
      if ( (unsigned int)v41 <= 0x40 )
        goto LABEL_10;
    }
  }
  if ( v40 )
    j_j___libc_free_0_0(v40);
LABEL_10:
  if ( !v19 )
    return 0;
  v14 = a1[2];
LABEL_12:
  v21 = *(_QWORD *)(*v14 + 40);
  *v14 = *(_QWORD *)v21;
  *((_DWORD *)v14 + 2) = *(_DWORD *)(v21 + 8);
  v22 = (unsigned __int16 *)(*(_QWORD *)(*a1[2] + 48) + 16LL * *((unsigned int *)a1[2] + 2));
  v23 = *v22;
  v24 = (const __m128i *)*((_QWORD *)v22 + 1);
  LOWORD(v37[0]) = v23;
  v37[1] = v24;
  if ( (_WORD)v23 )
  {
    if ( (unsigned __int16)(v23 - 17) > 0xD3u )
    {
      LOWORD(v40) = v23;
      v41 = v24;
      goto LABEL_15;
    }
    LOWORD(v23) = word_4456580[v23 - 1];
    v35 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)v37) )
    {
      v41 = v24;
      LOWORD(v40) = 0;
      goto LABEL_20;
    }
    LOWORD(v23) = sub_3009970((__int64)v37, v12, v26, v27, v28);
  }
  LOWORD(v40) = v23;
  v41 = v35;
  if ( (_WORD)v23 )
  {
LABEL_15:
    if ( (_WORD)v23 == 1 || (unsigned __int16)(v23 - 504) <= 7u )
      BUG();
    v25 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v23 - 16];
    goto LABEL_21;
  }
LABEL_20:
  LODWORD(v25) = sub_3007260((__int64)&v40);
LABEL_21:
  v29 = a1[4];
  v30 = **a1;
  v38 = 0;
  v36 = v30;
  v31 = (__m128i *)sub_22077B0(0x18u);
  if ( v31 )
  {
    v31->m128i_i32[2] = v25;
    v31[1].m128i_i64[0] = (__int64)v29;
    v31->m128i_i64[0] = v36;
  }
  v37[0] = v31;
  v39 = sub_325FFE0;
  v38 = sub_325EE30;
  v32 = a1[1];
  v33 = *v32;
  v34 = v32[1];
  v42 = 0;
  sub_325EE30(&v40, v37, 2);
  v43 = v39;
  v42 = v38;
  v16 = sub_33CAAD0(v33, v34, &v40, 0, 0);
  if ( v42 )
    v42(&v40, (const __m128i **)&v40, 3);
  if ( v38 )
    v38((unsigned __int64 *)v37, v37, 3);
  return v16;
}
