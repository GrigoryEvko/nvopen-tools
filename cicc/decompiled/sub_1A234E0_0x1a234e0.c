// Function: sub_1A234E0
// Address: 0x1a234e0
//
__int64 __fastcall sub_1A234E0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r14
  __int64 v12; // rbx
  __int64 v13; // r15
  char v14; // al
  unsigned int v15; // eax
  __int64 v16; // r9
  __m128i v18; // xmm0
  __int16 v19; // ax
  __int64 v20; // rdi
  int v21; // r12d
  unsigned int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // rax
  char v25; // al
  unsigned int v26; // ecx
  __int64 v27; // rsi
  unsigned __int64 v28; // rax
  unsigned int v29; // ecx
  unsigned __int64 v30; // rdx
  unsigned int v31; // ebx
  __int64 v32; // rax
  unsigned __int64 *v33; // rdx
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  unsigned __int64 *v37; // rdi
  unsigned __int64 *v38; // rdi
  unsigned int v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v41; // [rsp+8h] [rbp-98h]
  unsigned __int64 v43; // [rsp+18h] [rbp-88h]
  unsigned __int64 v44; // [rsp+18h] [rbp-88h]
  __int64 v45; // [rsp+18h] [rbp-88h]
  int v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+28h] [rbp-78h]
  __int64 v49; // [rsp+28h] [rbp-78h]
  __int64 v50; // [rsp+28h] [rbp-78h]
  unsigned __int64 v51; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v52; // [rsp+38h] [rbp-68h]
  unsigned __int64 *v53; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v54; // [rsp+48h] [rbp-58h]
  __m128i v55; // [rsp+50h] [rbp-50h] BYREF
  __int64 v56; // [rsp+60h] [rbp-40h]

  v9 = a5;
  v12 = a4;
  v13 = a7.m128i_i64[0];
  while ( *(_DWORD *)(v9 + 8) <= 0x40u )
  {
    if ( !*(_QWORD *)v9 )
      goto LABEL_10;
LABEL_4:
    v14 = *(_BYTE *)(v12 + 8);
    switch ( v14 )
    {
      case 15:
        return 0;
      case 16:
        v15 = sub_127FA20(a2, **(_QWORD **)(v12 + 16));
        v16 = 0;
        if ( (v15 & 7) != 0 )
          return v16;
        v26 = *(_DWORD *)(v9 + 8);
        v27 = v15 >> 3;
        v52 = v26;
        if ( v26 <= 0x40 )
          v51 = v27 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v26);
        else
          sub_16A4EF0((__int64)&v51, v27, 0);
        sub_16A9F90((__int64)&v53, v9, (__int64)&v51);
        v39 = v54;
        if ( v54 <= 0x40 )
        {
          v16 = 0;
          if ( *(_QWORD *)(v12 + 32) < (unsigned __int64)v53 )
            goto LABEL_25;
LABEL_45:
          sub_16A7B50((__int64)&v55, (__int64)&v53, (__int64 *)&v51);
          sub_16A7590(v9, v55.m128i_i64);
          sub_135E100(v55.m128i_i64);
          v55.m128i_i64[0] = sub_159C0E0(*(__int64 **)(a1 + 24), (__int64)&v53);
          sub_12A9700(a7.m128i_i64[0], &v55);
          v32 = sub_1A234E0(a1, a2, (_DWORD)a3, *(_QWORD *)(v12 + 24), v9, a6, *(_OWORD *)&a7, a8, a9);
LABEL_46:
          v16 = v32;
          if ( v54 > 0x40 )
          {
            v38 = v53;
            if ( v53 )
            {
LABEL_48:
              v50 = v16;
              j_j___libc_free_0_0(v38);
              v16 = v50;
            }
          }
LABEL_25:
          if ( v52 > 0x40 )
          {
            if ( v51 )
            {
              v48 = v16;
              j_j___libc_free_0_0(v51);
              return v48;
            }
          }
          return v16;
        }
        v43 = *(_QWORD *)(v12 + 32);
        if ( v39 - (unsigned int)sub_16A57B0((__int64)&v53) <= 0x40 )
        {
          v37 = v53;
          if ( v43 >= *v53 )
            goto LABEL_45;
          goto LABEL_51;
        }
LABEL_52:
        v38 = v53;
        v16 = 0;
        if ( v53 )
          goto LABEL_48;
        goto LABEL_25;
      case 14:
        v40 = *(_QWORD *)(v12 + 24);
        v28 = sub_12BE0A0(a2, v40);
        v29 = *(_DWORD *)(v9 + 8);
        v52 = v29;
        if ( v29 <= 0x40 )
          v51 = v28 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v29);
        else
          sub_16A4EF0((__int64)&v51, v28, 0);
        sub_16A9F90((__int64)&v53, v9, (__int64)&v51);
        v30 = *(_QWORD *)(v12 + 32);
        v31 = v54;
        if ( v54 > 0x40 )
        {
          v44 = v30;
          if ( v31 - (unsigned int)sub_16A57B0((__int64)&v53) > 0x40 )
            goto LABEL_52;
          v37 = v53;
          if ( v44 < *v53 )
          {
LABEL_51:
            j_j___libc_free_0_0(v37);
            v16 = 0;
            goto LABEL_25;
          }
        }
        else
        {
          v16 = 0;
          if ( v30 < (unsigned __int64)v53 )
            goto LABEL_25;
        }
        sub_16A7B50((__int64)&v55, (__int64)&v53, (__int64 *)&v51);
        sub_16A7590(v9, v55.m128i_i64);
        sub_135E100(v55.m128i_i64);
        v55.m128i_i64[0] = sub_159C0E0(*(__int64 **)(a1 + 24), (__int64)&v53);
        sub_12A9700(a7.m128i_i64[0], &v55);
        v32 = sub_1A234E0(a1, a2, (_DWORD)a3, v40, v9, a6, *(_OWORD *)&a7, a8, a9);
        goto LABEL_46;
    }
    if ( v14 != 13 )
      return 0;
    v33 = (unsigned __int64 *)sub_15A9930(a2, v12);
    v34 = *(_DWORD *)(v9 + 8) <= 0x40u ? *(_QWORD *)v9 : **(_QWORD **)v9;
    if ( *v33 <= v34 )
      return 0;
    v41 = v33;
    v49 = (unsigned int)sub_15A8020((__int64)v33, v34);
    sub_135E0D0((__int64)&v55, *(_DWORD *)(v9 + 8), v41[v49 + 2], 0);
    sub_16A7590(v9, v55.m128i_i64);
    sub_135E100(v55.m128i_i64);
    v12 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 8 * v49);
    v35 = sub_12BE0A0(a2, v12);
    if ( !sub_13D0480(v9, v35) )
      return 0;
    v36 = sub_1643350(*(_QWORD **)(a1 + 24));
    v55.m128i_i64[0] = sub_159C470(v36, v49, 0);
    sub_12A9700(a7.m128i_i64[0], &v55);
  }
  v47 = *(_DWORD *)(v9 + 8);
  if ( v47 - (unsigned int)sub_16A57B0(v9) > 0x40 || **(_QWORD **)v9 )
    goto LABEL_4;
LABEL_10:
  v18 = _mm_loadu_si128((const __m128i *)&a7.m128i_u64[1]);
  v19 = a9;
  v55 = v18;
  v56 = a9;
  if ( v12 == a6 )
  {
    a7 = v18;
  }
  else
  {
    v20 = a2;
    v21 = 0;
    v22 = sub_15A9570(v20, *a3);
    while ( 1 )
    {
      v25 = *(_BYTE *)(v12 + 8);
      if ( v25 == 15 )
        break;
      if ( v25 == 14 )
      {
        v12 = *(_QWORD *)(v12 + 24);
        v23 = sub_1644C60(*(_QWORD **)(a1 + 24), v22);
      }
      else
      {
        if ( v25 == 16 )
        {
          v12 = *(_QWORD *)(v12 + 24);
        }
        else
        {
          if ( v25 != 13 || !*(_DWORD *)(v12 + 12) )
            break;
          v12 = **(_QWORD **)(v12 + 16);
        }
        v23 = sub_1643350(*(_QWORD **)(a1 + 24));
      }
      a5 = sub_159C470(v23, 0, 0);
      v24 = *(unsigned int *)(a7.m128i_i64[0] + 8);
      if ( (unsigned int)v24 >= *(_DWORD *)(a7.m128i_i64[0] + 12) )
      {
        v45 = a5;
        sub_16CD150(a7.m128i_i64[0], (const void *)(a7.m128i_i64[0] + 16), 0, 8, a5, a6);
        v24 = *(unsigned int *)(a7.m128i_i64[0] + 8);
        a5 = v45;
      }
      ++v21;
      *(_QWORD *)(*(_QWORD *)a7.m128i_i64[0] + 8 * v24) = a5;
      ++*(_DWORD *)(a7.m128i_i64[0] + 8);
      if ( a6 == v12 )
        goto LABEL_62;
    }
    if ( a6 != v12 )
      *(_DWORD *)(a7.m128i_i64[0] + 8) -= v21;
LABEL_62:
    v19 = v56;
    a7 = _mm_loadu_si128(&v55);
  }
  return sub_1A1DA80((__int64 *)a1, a3, v13, a4, a5, a6, (__int64 *)a7.m128i_i64[0], a7.m128i_i32[2], v19);
}
