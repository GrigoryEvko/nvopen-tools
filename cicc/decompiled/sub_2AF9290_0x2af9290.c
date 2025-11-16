// Function: sub_2AF9290
// Address: 0x2af9290
//
__int64 __fastcall sub_2AF9290(__int64 a1, char **a2, __int64 a3)
{
  char **v3; // r12
  char *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r9
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  _BYTE *v16; // r8
  __int64 *v17; // rdx
  char *v18; // r15
  __int64 *v19; // rbx
  unsigned __int8 *v20; // r12
  char v21; // al
  __int64 v22; // r13
  __int64 v23; // r14
  _QWORD *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // rax
  char v30; // r13
  int v31; // eax
  __int64 v32; // rax
  __int64 *v33; // r15
  __int64 *v34; // r12
  char v35; // di
  __int64 v36; // rsi
  _QWORD *v37; // rax
  __int64 v38; // r13
  __int64 v39; // r14
  __int64 v40; // rsi
  _QWORD *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r14
  __int64 *v45; // rax
  int v46; // [rsp+10h] [rbp-4A0h]
  __int64 v48; // [rsp+20h] [rbp-490h]
  __int64 v49; // [rsp+28h] [rbp-488h]
  __int64 v50; // [rsp+30h] [rbp-480h]
  bool v51; // [rsp+4Eh] [rbp-462h]
  char v52; // [rsp+4Fh] [rbp-461h]
  __int64 *v53; // [rsp+50h] [rbp-460h]
  __int64 v54; // [rsp+58h] [rbp-458h]
  __int64 v55; // [rsp+60h] [rbp-450h] BYREF
  __int64 v56; // [rsp+70h] [rbp-440h]
  __m128i v57; // [rsp+80h] [rbp-430h] BYREF
  __int64 v58; // [rsp+90h] [rbp-420h]
  __int64 v59; // [rsp+98h] [rbp-418h]
  __int64 v60; // [rsp+A0h] [rbp-410h]
  __int64 v61; // [rsp+A8h] [rbp-408h]
  char v62; // [rsp+B0h] [rbp-400h]
  __int64 *v63; // [rsp+C0h] [rbp-3F0h] BYREF
  __int64 v64; // [rsp+C8h] [rbp-3E8h]
  _BYTE v65[128]; // [rsp+D0h] [rbp-3E0h] BYREF
  _BYTE *v66; // [rsp+150h] [rbp-360h] BYREF
  __int64 v67; // [rsp+158h] [rbp-358h]
  _BYTE v68[128]; // [rsp+160h] [rbp-350h] BYREF
  __m128i v69; // [rsp+1E0h] [rbp-2D0h] BYREF
  __int64 v70; // [rsp+1F0h] [rbp-2C0h]
  __int64 v71; // [rsp+1F8h] [rbp-2B8h] BYREF
  _QWORD v72[39]; // [rsp+200h] [rbp-2B0h] BYREF
  _QWORD v73[2]; // [rsp+338h] [rbp-178h] BYREF
  char v74; // [rsp+348h] [rbp-168h]
  _BYTE *v75; // [rsp+350h] [rbp-160h]
  __int64 v76; // [rsp+358h] [rbp-158h]
  _BYTE v77[128]; // [rsp+360h] [rbp-150h] BYREF
  __int16 v78; // [rsp+3E0h] [rbp-D0h]
  _QWORD v79[2]; // [rsp+3E8h] [rbp-C8h] BYREF
  __int64 v80; // [rsp+3F8h] [rbp-B8h]
  __int64 v81; // [rsp+400h] [rbp-B0h] BYREF
  unsigned int v82; // [rsp+408h] [rbp-A8h]
  char v83; // [rsp+480h] [rbp-30h] BYREF

  v3 = &a2[a3];
  v63 = (__int64 *)v65;
  v64 = 0x1000000000LL;
  v67 = 0x1000000000LL;
  v4 = *a2;
  v66 = v68;
  v46 = a3;
  v52 = *v4;
  v51 = *v4 == 61;
  sub_2AF7570((__int64)&v55, a2, a3);
  v7 = v55;
  v8 = v56;
  if ( v56 != v55 )
  {
    v54 = (__int64)&v63;
    do
    {
      while ( 1 )
      {
        if ( !v7 )
          BUG();
        v11 = v7 - 24;
        if ( (unsigned __int8)(*(_BYTE *)(v7 - 24) - 61) <= 1u )
        {
          v69.m128i_i64[0] = v7 - 24;
          if ( v3 != sub_2AF7070(a2, (__int64)v3, v69.m128i_i64) )
            break;
        }
        if ( (unsigned __int8)sub_98CD80((char *)(v7 - 24)) )
        {
          if ( (unsigned __int8)sub_B46420(v7 - 24) || (unsigned __int8)sub_B46490(v7 - 24) )
          {
            v9 = (unsigned int)v64;
            v5 = HIDWORD(v64);
            v10 = (unsigned int)v64 + 1LL;
            if ( v10 > HIDWORD(v64) )
            {
              sub_C8D5F0((__int64)&v63, v65, v10, 8u, v13, v6);
              v9 = (unsigned int)v64;
            }
            v63[v9] = v11;
            LODWORD(v64) = v64 + 1;
          }
          v7 = *(_QWORD *)(v7 + 8);
          if ( v8 != v7 )
            continue;
        }
        goto LABEL_17;
      }
      v14 = (unsigned int)v67;
      v5 = HIDWORD(v67);
      v15 = (unsigned int)v67 + 1LL;
      if ( v15 > HIDWORD(v67) )
      {
        sub_C8D5F0((__int64)&v66, v68, v15, 8u, v12, v6);
        v14 = (unsigned int)v67;
      }
      *(_QWORD *)&v66[8 * v14] = v11;
      LODWORD(v67) = v67 + 1;
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v8 != v7 );
  }
LABEL_17:
  if ( !(_DWORD)v67 )
  {
    v69.m128i_i64[0] = 0;
    v69.m128i_i64[1] = (__int64)v72;
    v70 = 8;
    LODWORD(v71) = 0;
    BYTE4(v71) = 1;
    LODWORD(v54) = v46;
    if ( !v46 )
    {
      v16 = v66;
      v43 = 0;
      goto LABEL_83;
    }
    v35 = 1;
LABEL_70:
    v38 = 0;
    v39 = 0xFFFFFFFFLL;
    while ( 1 )
    {
      v40 = (__int64)a2[v38];
      if ( v35 )
      {
        v41 = (_QWORD *)v69.m128i_i64[1];
        v42 = v69.m128i_i64[1] + 8LL * HIDWORD(v70);
        if ( v69.m128i_i64[1] == v42 )
          goto LABEL_89;
        while ( v40 != *v41 )
        {
          if ( (_QWORD *)v42 == ++v41 )
            goto LABEL_89;
        }
      }
      else
      {
        v45 = sub_C8CA60((__int64)&v69, v40);
        v35 = BYTE4(v71);
        if ( !v45 )
        {
LABEL_89:
          if ( (_DWORD)v39 != -1 )
          {
            LODWORD(v54) = v38 - v39;
            goto LABEL_81;
          }
          goto LABEL_78;
        }
      }
      if ( (_DWORD)v39 == -1 )
        v39 = (unsigned int)v38;
LABEL_78:
      if ( ++v38 == (unsigned int)v54 )
      {
        if ( (_DWORD)v39 != -1 )
        {
          LODWORD(v54) = v54 - v39;
          goto LABEL_81;
        }
        LODWORD(v54) = 0;
        goto LABEL_99;
      }
    }
  }
  v16 = v66;
  v48 = (unsigned int)v67;
  v50 = 0;
  v54 = 0;
  v17 = (__int64 *)v66;
  while ( 1 )
  {
    v18 = *(char **)&v16[8 * v50];
    v49 = 8 * v50;
    if ( !v54 )
    {
      v19 = v63;
      v53 = &v63[(unsigned int)v64];
      if ( v53 != v63 )
        goto LABEL_25;
      goto LABEL_48;
    }
    if ( sub_B445A0(v54, (__int64)v18) )
      break;
    v19 = v63;
    v53 = &v63[(unsigned int)v64];
    if ( v53 != v63 )
    {
      do
      {
        while ( 1 )
        {
LABEL_25:
          v20 = (unsigned __int8 *)*v19;
          if ( v54 && sub_B445A0(v54, *v19) )
            goto LABEL_56;
          v21 = *v18;
          if ( *v20 != 61 )
            break;
          if ( v21 == 61 )
            goto LABEL_24;
          v22 = 0;
          v23 = (__int64)v20;
          if ( v52 != 61 )
            goto LABEL_31;
LABEL_51:
          if ( sub_B445A0((__int64)v18, (__int64)v20) )
            goto LABEL_24;
          if ( !v22 || (*(_BYTE *)(v22 + 7) & 0x20) == 0 || !sub_B91C10(v22, 6) )
            goto LABEL_35;
          if ( v53 == ++v19 )
            goto LABEL_56;
        }
        v22 = 0;
        if ( v21 == 61 )
          v22 = (__int64)v18;
        v23 = 0;
        if ( v52 == 61 )
          goto LABEL_51;
LABEL_31:
        if ( sub_B445A0((__int64)v20, (__int64)v18) || v23 && (*(_BYTE *)(v23 + 7) & 0x20) != 0 && sub_B91C10(v23, 6) )
          goto LABEL_24;
LABEL_35:
        v24 = *(_QWORD **)(a1 + 8);
        sub_D66840(&v69, v18);
        v25 = v69.m128i_i64[0];
        v62 = 1;
        v69.m128i_i64[0] = (__int64)v24;
        v57.m128i_i64[0] = v25;
        v26 = v69.m128i_i64[1];
        v69.m128i_i64[1] = 0;
        v57.m128i_i64[1] = v26;
        v27 = v70;
        v70 = 1;
        v58 = v27;
        v59 = v71;
        v60 = v72[0];
        v61 = v72[1];
        v28 = &v71;
        do
        {
          *v28 = -4;
          v28 += 5;
          *(v28 - 4) = -3;
          *(v28 - 3) = -4;
          *(v28 - 2) = -3;
        }
        while ( v28 != v73 );
        v73[1] = 0;
        v75 = v77;
        v76 = 0x400000000LL;
        v78 = 256;
        v73[0] = v79;
        v74 = 0;
        v79[1] = 0;
        v80 = 1;
        v79[0] = &unk_49DDBE8;
        v29 = &v81;
        do
        {
          *v29 = -4096;
          v29 += 2;
        }
        while ( v29 != (__int64 *)&v83 );
        v30 = sub_CF63E0(v24, v20, &v57, (__int64)&v69);
        v79[0] = &unk_49DDBE8;
        if ( (v80 & 1) == 0 )
          sub_C7D6A0(v81, 16LL * v82, 8);
        nullsub_184();
        if ( v75 != v77 )
          _libc_free((unsigned __int64)v75);
        if ( (v70 & 1) == 0 )
          sub_C7D6A0(v71, 40LL * LODWORD(v72[0]), 8);
        if ( v52 == 61 )
        {
          if ( (v30 & 2) != 0 )
            goto LABEL_60;
        }
        else if ( v30 )
        {
          v16 = v66;
          v54 = (__int64)v20;
          v17 = (__int64 *)v66;
          goto LABEL_48;
        }
LABEL_24:
        ++v19;
      }
      while ( v53 != v19 );
    }
LABEL_56:
    v16 = v66;
    v17 = (__int64 *)v66;
    if ( v54 && v51 )
    {
      v33 = (__int64 *)v66;
      goto LABEL_61;
    }
LABEL_48:
    v31 = v50;
    v5 = v50 + 1;
    v50 = v5;
    v32 = (unsigned int)(v31 + 1);
    if ( v5 == v48 )
    {
      v33 = v17;
      v49 = 8 * v32;
      goto LABEL_61;
    }
  }
LABEL_60:
  v16 = v66;
  v33 = (__int64 *)v66;
LABEL_61:
  v69.m128i_i64[0] = 0;
  v69.m128i_i64[1] = (__int64)v72;
  v34 = (__int64 *)&v16[v49];
  BYTE4(v71) = 1;
  v70 = 8;
  LODWORD(v71) = 0;
  if ( &v16[v49] == v16 )
  {
    v35 = 1;
    v43 = 0;
    LODWORD(v54) = v46;
    if ( !v46 )
      goto LABEL_83;
    goto LABEL_70;
  }
  v35 = 1;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v36 = *v33;
        if ( v35 )
          break;
LABEL_93:
        ++v33;
        sub_C8CC70((__int64)&v69, v36, (__int64)v17, v5, (__int64)v16, v6);
        v35 = BYTE4(v71);
        if ( v34 == v33 )
          goto LABEL_69;
      }
      v37 = (_QWORD *)v69.m128i_i64[1];
      v5 = HIDWORD(v70);
      v17 = (__int64 *)(v69.m128i_i64[1] + 8LL * HIDWORD(v70));
      if ( (__int64 *)v69.m128i_i64[1] != v17 )
        break;
LABEL_95:
      if ( HIDWORD(v70) >= (unsigned int)v70 )
        goto LABEL_93;
      v5 = (unsigned int)(HIDWORD(v70) + 1);
      ++v33;
      ++HIDWORD(v70);
      *v17 = v36;
      v35 = BYTE4(v71);
      ++v69.m128i_i64[0];
      if ( v34 == v33 )
        goto LABEL_69;
    }
    while ( v36 != *v37 )
    {
      if ( v17 == ++v37 )
        goto LABEL_95;
    }
    ++v33;
  }
  while ( v34 != v33 );
LABEL_69:
  LODWORD(v54) = v46;
  if ( v46 )
    goto LABEL_70;
LABEL_99:
  v39 = 0;
LABEL_81:
  v16 = v66;
  v43 = (v54 << 32) | v39;
  if ( !v35 )
  {
    _libc_free(v69.m128i_u64[1]);
    v16 = v66;
  }
LABEL_83:
  if ( v16 != v68 )
    _libc_free((unsigned __int64)v16);
  if ( v63 != (__int64 *)v65 )
    _libc_free((unsigned __int64)v63);
  return v43;
}
