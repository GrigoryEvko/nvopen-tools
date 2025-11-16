// Function: sub_1195590
// Address: 0x1195590
//
__int64 __fastcall sub_1195590(__int64 a1, unsigned __int8 *a2, const __m128i *a3, char a4)
{
  __int64 v5; // r13
  unsigned __int8 *v8; // rdx
  unsigned __int8 *v9; // r14
  unsigned __int8 v10; // al
  _BYTE *v11; // r10
  unsigned __int8 *v12; // r15
  __int64 *v13; // rdx
  _BYTE *v14; // r11
  char v15; // r9
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v19; // xmm0
  __m128i v20; // xmm1
  unsigned __int64 v21; // xmm2_8
  unsigned __int8 *v22; // rsi
  __m128i v23; // xmm3
  unsigned __int8 *v24; // rax
  unsigned int v25; // eax
  __int64 v26; // r11
  char v27; // r9
  __int64 v28; // rdi
  char v29; // al
  __int64 v30; // r11
  char v31; // r9
  char v32; // dl
  __int64 v33; // rdi
  char v34; // al
  char v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // r14
  __int64 v39; // rbx
  __int64 v40; // r14
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rdx
  __int64 v44; // rdx
  _BYTE *v45; // rax
  unsigned __int8 *v46; // r11
  __int64 v47; // rdx
  __int64 v48; // rdx
  _BOOL4 v49; // esi
  __int64 v50; // rdx
  _BYTE *v51; // rax
  char v52; // r14
  unsigned __int8 *v53; // r15
  unsigned int v54; // ebx
  _BYTE *v55; // rax
  char v56; // si
  char v57; // si
  unsigned __int8 *v58; // r14
  unsigned int v59; // ebx
  _BYTE *v60; // rax
  char v61; // [rsp+Ch] [rbp-D4h]
  const __m128i *v62; // [rsp+10h] [rbp-D0h]
  __int64 v63; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v64; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v65; // [rsp+20h] [rbp-C0h]
  char v66; // [rsp+20h] [rbp-C0h]
  char v67; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *v68; // [rsp+20h] [rbp-C0h]
  char v69; // [rsp+20h] [rbp-C0h]
  char v70; // [rsp+20h] [rbp-C0h]
  unsigned __int32 v71; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v72; // [rsp+28h] [rbp-B8h]
  const __m128i *v73; // [rsp+28h] [rbp-B8h]
  unsigned int v74; // [rsp+30h] [rbp-B0h]
  char v75; // [rsp+37h] [rbp-A9h]
  int v77; // [rsp+38h] [rbp-A8h]
  char v78; // [rsp+38h] [rbp-A8h]
  char v79; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v80; // [rsp+38h] [rbp-A8h]
  char v81; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v82; // [rsp+40h] [rbp-A0h]
  char v83; // [rsp+40h] [rbp-A0h]
  __int64 v84; // [rsp+40h] [rbp-A0h]
  int v85; // [rsp+40h] [rbp-A0h]
  __int64 v86; // [rsp+40h] [rbp-A0h]
  __int64 v87; // [rsp+40h] [rbp-A0h]
  int v88; // [rsp+40h] [rbp-A0h]
  __int64 v89; // [rsp+40h] [rbp-A0h]
  bool v90; // [rsp+48h] [rbp-98h]
  __int64 v91; // [rsp+48h] [rbp-98h]
  __int64 v92; // [rsp+48h] [rbp-98h]
  __int64 v93; // [rsp+48h] [rbp-98h]
  int v94; // [rsp+48h] [rbp-98h]
  __int64 v95; // [rsp+48h] [rbp-98h]
  unsigned __int8 *v96; // [rsp+50h] [rbp-90h]
  __m128i v98[2]; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v99; // [rsp+80h] [rbp-60h]
  unsigned __int8 *v100; // [rsp+88h] [rbp-58h]
  __m128i v101; // [rsp+90h] [rbp-50h]
  __int64 v102; // [rsp+A0h] [rbp-40h]

  if ( !a2 || (unsigned int)*a2 - 54 > 2 )
    return 0;
  if ( (a2[7] & 0x40) != 0 )
  {
    v8 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    v9 = *(unsigned __int8 **)v8;
    v10 = **(_BYTE **)v8;
    if ( v10 <= 0x1Cu )
      return 0;
  }
  else
  {
    v8 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v9 = *(unsigned __int8 **)v8;
    v10 = **(_BYTE **)v8;
    if ( v10 <= 0x1Cu )
      return 0;
  }
  v11 = (_BYTE *)*((_QWORD *)v8 + 4);
  if ( *v11 == 68 && *((_QWORD *)v11 - 4) )
    v11 = (_BYTE *)*((_QWORD *)v11 - 4);
  if ( v10 != 67 || (v12 = (unsigned __int8 *)*((_QWORD *)v9 - 4), *v12 <= 0x1Cu) )
  {
    v12 = v9;
    v9 = 0;
  }
  if ( (unsigned int)*v12 - 54 > 2 )
    return 0;
  if ( (v12[7] & 0x40) != 0 )
  {
    v13 = (__int64 *)*((_QWORD *)v12 - 1);
    v5 = *v13;
    if ( !*v13 )
      return 0;
  }
  else
  {
    v13 = (__int64 *)&v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
    v5 = *v13;
    if ( !*v13 )
      return 0;
  }
  v14 = (_BYTE *)v13[4];
  if ( *v14 == 68 && *((_QWORD *)v14 - 4) )
    v14 = (_BYTE *)*((_QWORD *)v14 - 4);
  v82 = v14;
  v96 = v11;
  v90 = sub_11954A0((__int64)a2, (__int64)v11, (__int64)v12, (__int64)v14);
  v15 = a4;
  if ( !v90 )
    return 0;
  v77 = *a2;
  if ( (unsigned int)(v77 - 55) > 1 || (v16 = *v12, (unsigned int)(v16 - 55) > 1) )
  {
    if ( v15 )
      return 0;
    v90 = 0;
    LOBYTE(v16) = *v12;
  }
  v75 = (v15 ^ 1) & (*a2 != (unsigned __int8)v16);
  if ( v75 )
    return 0;
  if ( v9 )
  {
    if ( !v15 )
    {
      v17 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 16LL);
      if ( !v17 || *(_QWORD *)(v17 + 8) )
      {
        v18 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 16LL);
        if ( !v18 || *(_QWORD *)(v18 + 8) )
          return 0;
      }
    }
  }
  v19 = _mm_loadu_si128(a3);
  v20 = _mm_loadu_si128(a3 + 1);
  v21 = _mm_loadu_si128(a3 + 2).m128i_u64[0];
  v22 = v82;
  v102 = a3[4].m128i_i64[0];
  v23 = _mm_loadu_si128(a3 + 3);
  v83 = v15;
  v99 = v21;
  v100 = a2;
  v98[0] = v19;
  v98[1] = v20;
  v101 = v23;
  v24 = sub_101BE10(v96, v22, 0, 0, v98);
  if ( !v24 || *v24 > 0x15u )
    return 0;
  v65 = v24;
  v71 = sub_BCB060(*((_QWORD *)v24 + 1));
  v25 = sub_BCB060(*(_QWORD *)(v5 + 8));
  v26 = (__int64)v65;
  v27 = v83;
  v74 = v25;
  v98[0].m128i_i32[2] = v71;
  if ( v71 > 0x40 )
  {
    v67 = v83;
    v87 = v26;
    sub_C43690((__int64)v98, v25, 0);
    v27 = v67;
    v26 = v87;
  }
  else
  {
    v98[0].m128i_i64[0] = v25;
  }
  v66 = v27;
  if ( *(_BYTE *)v26 == 17 )
  {
    v84 = v26;
    v28 = v26 + 24;
  }
  else
  {
    v44 = *(_QWORD *)(v26 + 8);
    v86 = v44;
    if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 > 1 )
      goto LABEL_54;
    v63 = v26;
    v45 = sub_AD7630(v26, 0, v44);
    v46 = (unsigned __int8 *)v63;
    if ( !v45 || *v45 != 17 )
    {
      if ( *(_BYTE *)(v86 + 8) == 17 )
      {
        v88 = *(_DWORD *)(v86 + 32);
        if ( v88 )
        {
          v61 = v66;
          v68 = v9;
          v52 = 0;
          v64 = v12;
          v53 = v46;
          v62 = a3;
          v54 = 0;
          while ( 1 )
          {
            v55 = (_BYTE *)sub_AD69F0(v53, v54);
            if ( !v55 )
              break;
            if ( *v55 != 13 )
            {
              if ( *v55 != 17 )
                break;
              v52 = sub_B532C0((__int64)(v55 + 24), v98, 36);
              if ( !v52 )
                break;
            }
            if ( v88 == ++v54 )
            {
              v30 = (__int64)v53;
              v32 = v52;
              v12 = v64;
              v9 = v68;
              a3 = v62;
              v31 = v61;
              goto LABEL_31;
            }
          }
        }
      }
      goto LABEL_54;
    }
    v84 = v63;
    v28 = (__int64)(v45 + 24);
  }
  v29 = sub_B532C0(v28, v98, 36);
  v30 = v84;
  v31 = v66;
  v32 = v29;
LABEL_31:
  if ( !v32 )
    goto LABEL_54;
  if ( v98[0].m128i_i32[2] > 0x40u && v98[0].m128i_i64[0] )
  {
    v70 = v31;
    v89 = v30;
    j_j___libc_free_0_0(v98[0].m128i_i64[0]);
    v31 = v70;
    v30 = v89;
  }
  v85 = v77 - 29;
  if ( !v90 )
  {
LABEL_44:
    v36 = *(_QWORD *)(v5 + 8);
    if ( v36 == *(_QWORD *)(v30 + 8) || (v30 = sub_96F480(0x27u, v30, v36, a3->m128i_i64[0])) != 0 )
    {
      LOWORD(v99) = 257;
      v37 = sub_B504D0(v85, v5, v30, (__int64)v98, 0, 0);
      v5 = v37;
      if ( v9 )
      {
        LOWORD(v99) = 257;
        v38 = *(__int64 **)(a1 + 32);
        (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v38[11] + 16LL))(
          v38[11],
          v37,
          v98,
          v38[7],
          v38[8]);
        v39 = *v38;
        v40 = *v38 + 16LL * *((unsigned int *)v38 + 2);
        while ( v40 != v39 )
        {
          v41 = *(_QWORD *)(v39 + 8);
          v42 = *(_DWORD *)v39;
          v39 += 16;
          sub_B99FD0(v5, v42, v41);
        }
        v43 = *((_QWORD *)a2 + 1);
        LOWORD(v99) = 257;
        return sub_B51D30(38, v5, v43, (__int64)v98, 0, 0);
      }
      goto LABEL_71;
    }
    return 0;
  }
  if ( !v9 && !v31 )
  {
    v47 = *(_QWORD *)(v5 + 8);
    if ( *(_QWORD *)(v30 + 8) == v47 )
    {
      LOWORD(v99) = 257;
      v5 = sub_B504D0(v85, v5, v30, (__int64)v98, 0, 0);
    }
    else
    {
      v48 = sub_96F480(0x27u, v30, v47, a3->m128i_i64[0]);
      if ( !v48 )
        return 0;
      LOWORD(v99) = 257;
      v5 = sub_B504D0(v85, v5, v48, (__int64)v98, 0, 0);
    }
LABEL_71:
    if ( v85 == 25 )
    {
      v56 = 0;
      if ( sub_B448F0((__int64)a2) )
        v56 = sub_B448F0((__int64)v12);
      sub_B447F0((unsigned __int8 *)v5, v56);
      v57 = 0;
      if ( sub_B44900((__int64)a2) )
        v57 = sub_B44900((__int64)v12);
      sub_B44850((unsigned __int8 *)v5, v57);
    }
    else
    {
      v49 = 0;
      if ( sub_B44E60((__int64)a2) )
        v49 = sub_B44E60((__int64)v12);
      sub_B448B0(v5, v49);
    }
    return v5;
  }
  v98[0].m128i_i32[2] = v71;
  if ( v71 > 0x40 )
  {
    v79 = v31;
    v93 = v30;
    sub_C43690((__int64)v98, v74 - 1, 0);
    v31 = v79;
    v30 = v93;
  }
  else
  {
    v98[0].m128i_i64[0] = v74 - 1;
  }
  v78 = v31;
  if ( *(_BYTE *)v30 == 17 )
  {
    v91 = v30;
    v33 = v30 + 24;
    goto LABEL_40;
  }
  v50 = *(_QWORD *)(v30 + 8);
  v92 = v50;
  if ( (unsigned int)*(unsigned __int8 *)(v50 + 8) - 17 > 1 )
  {
LABEL_54:
    if ( v98[0].m128i_i32[2] > 0x40u && v98[0].m128i_i64[0] )
      j_j___libc_free_0_0(v98[0].m128i_i64[0]);
    return 0;
  }
  v72 = (unsigned __int8 *)v30;
  v51 = sub_AD7630(v30, 0, v50);
  if ( !v51 || *v51 != 17 )
  {
    if ( *(_BYTE *)(v92 + 8) == 17 )
    {
      v94 = *(_DWORD *)(v92 + 32);
      if ( v94 )
      {
        v69 = v78;
        v80 = v9;
        v58 = v72;
        v73 = a3;
        v59 = 0;
        while ( 1 )
        {
          v60 = (_BYTE *)sub_AD69F0(v58, v59);
          if ( !v60 )
            break;
          if ( *v60 != 13 )
          {
            if ( *v60 != 17 )
              break;
            v75 = sub_B532C0((__int64)(v60 + 24), v98, 32);
            if ( !v75 )
              break;
          }
          if ( v94 == ++v59 )
          {
            v30 = (__int64)v58;
            a3 = v73;
            v9 = v80;
            v35 = v69;
            goto LABEL_41;
          }
        }
      }
    }
    goto LABEL_54;
  }
  v91 = (__int64)v72;
  v33 = (__int64)(v51 + 24);
LABEL_40:
  v34 = sub_B532C0(v33, v98, 32);
  v30 = v91;
  v35 = v78;
  v75 = v34;
LABEL_41:
  if ( !v75 )
    goto LABEL_54;
  if ( v98[0].m128i_i32[2] > 0x40u && v98[0].m128i_i64[0] )
  {
    v81 = v35;
    v95 = v30;
    j_j___libc_free_0_0(v98[0].m128i_i64[0]);
    v30 = v95;
    v35 = v81;
  }
  if ( !v35 )
    goto LABEL_44;
  return v5;
}
