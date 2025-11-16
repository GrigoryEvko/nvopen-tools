// Function: sub_A12540
// Address: 0xa12540
//
__int64 *__fastcall sub_A12540(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r15
  __int64 *v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int64 v12; // rax
  __int64 *v13; // rsi
  __int64 v14; // rdi
  __int64 *v15; // rbx
  unsigned __int64 v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rsi
  int v19; // edx
  unsigned __int64 v21; // rax
  unsigned int v22; // r15d
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // esi
  int *v27; // rbx
  int v28; // r9d
  unsigned __int64 v29; // rcx
  _BYTE *v30; // rdx
  const char *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rax
  int v37; // r8d
  __int64 v38; // rax
  int v39; // [rsp+0h] [rbp-330h]
  __int64 *v40; // [rsp+8h] [rbp-328h]
  __int64 v41; // [rsp+10h] [rbp-320h]
  __int64 *v42; // [rsp+18h] [rbp-318h]
  int v43; // [rsp+20h] [rbp-310h]
  int v44; // [rsp+24h] [rbp-30Ch]
  unsigned int v45; // [rsp+28h] [rbp-308h]
  int v46; // [rsp+2Ch] [rbp-304h]
  unsigned __int64 v49; // [rsp+50h] [rbp-2E0h] BYREF
  __int64 v50; // [rsp+58h] [rbp-2D8h]
  __int64 v51; // [rsp+60h] [rbp-2D0h]
  __int64 v52; // [rsp+68h] [rbp-2C8h]
  unsigned __int64 v53; // [rsp+70h] [rbp-2C0h] BYREF
  __int64 v54; // [rsp+78h] [rbp-2B8h]
  __int64 v55; // [rsp+80h] [rbp-2B0h]
  unsigned __int64 v56; // [rsp+88h] [rbp-2A8h]
  char v57; // [rsp+90h] [rbp-2A0h]
  char v58; // [rsp+91h] [rbp-29Fh]
  __int64 v59; // [rsp+A0h] [rbp-290h] BYREF
  __int64 v60; // [rsp+A8h] [rbp-288h]
  const char *v61; // [rsp+B0h] [rbp-280h]
  __int64 v62; // [rsp+B8h] [rbp-278h]
  __int64 v63; // [rsp+C0h] [rbp-270h]
  unsigned __int64 v64; // [rsp+C8h] [rbp-268h]
  __int64 v65; // [rsp+D0h] [rbp-260h]
  __int64 v66; // [rsp+D8h] [rbp-258h]
  __int64 v67; // [rsp+E0h] [rbp-250h]
  __int64 v68; // [rsp+E8h] [rbp-248h]
  __int64 v69; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v70; // [rsp+F8h] [rbp-238h]
  _BYTE v71[560]; // [rsp+100h] [rbp-230h] BYREF

  v4 = &v69;
  sub_A4DCE0(&v69, a2[15].m128i_i64[0], 16, 0);
  if ( (v69 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v69 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v59 = 0;
  v7 = (__int64 *)&v53;
  v69 = (__int64)v71;
  v70 = 0x4000000000LL;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  sub_A04120(&v59, 0);
  while ( 1 )
  {
    do
    {
LABEL_3:
      sub_9CEFB0((__int64)v7, a2[15].m128i_i64[0], 0, v8);
      if ( (v54 & 1) != 0 )
      {
        LOBYTE(v54) = v54 & 0xFD;
        v21 = v53;
        v53 = 0;
        v49 = v21 | 1;
      }
      else
      {
        v49 = 1;
        v45 = HIDWORD(v53);
        v46 = v53;
      }
      v12 = v49 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v49 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_6;
      if ( v46 == 1 )
      {
        sub_A10370(a2, (__int64)&v59, v9, v10, v11);
        *a1 = 1;
        goto LABEL_7;
      }
      if ( (v46 & 0xFFFFFFFD) == 0 )
      {
        v58 = 1;
        v53 = (unsigned __int64)"Malformed block";
        v57 = 3;
        sub_A01DB0(a1, (__int64)v7);
        goto LABEL_7;
      }
      v18 = a2[15].m128i_i64[0];
      LODWORD(v70) = 0;
      sub_A4B600(&v49, v18, v45, v4, 0);
      v19 = v50 & 1;
      v8 = (unsigned int)(2 * v19);
      LOBYTE(v50) = (2 * v19) | v50 & 0xFD;
      if ( (_BYTE)v19 )
      {
        v12 = v49;
LABEL_6:
        *a1 = v12 | 1;
        goto LABEL_7;
      }
    }
    while ( (_DWORD)v49 != 11 );
    if ( !(_DWORD)v70 )
      break;
    v8 = v69;
    v44 = v70 & 1;
    if ( (v70 & 1) != 0 )
    {
      v41 = *(_QWORD *)(a4 + 8LL * *(_QWORD *)v69);
      if ( (_DWORD)v70 == 1 )
        goto LABEL_3;
      v43 = v70;
      v42 = v7;
      v40 = v4;
      v22 = v70 & 1;
      while ( 1 )
      {
        v23 = a2[67].m128i_i64[0];
        v24 = *(_QWORD *)(v8 + 8LL * v22);
        v25 = a2[68].m128i_u32[0];
        if ( !(_DWORD)v25 )
          goto LABEL_59;
        v26 = (v25 - 1) & (37 * v24);
        v27 = (int *)(v23 + 8LL * v26);
        v28 = *v27;
        if ( (_DWORD)v24 != *v27 )
        {
          v37 = v44;
          while ( v28 != -1 )
          {
            v26 = (v25 - 1) & (v26 + v37);
            v27 = (int *)(v23 + 8LL * v26);
            v28 = *v27;
            if ( (unsigned int)*(_QWORD *)(v8 + 8LL * v22) == *v27 )
              goto LABEL_35;
            ++v37;
          }
LABEL_59:
          v58 = 1;
          v7 = v42;
          v31 = "Invalid ID";
          goto LABEL_43;
        }
LABEL_35:
        if ( v27 == (int *)(v23 + 8 * v25) )
          goto LABEL_59;
        if ( v27[1] != 1 || !a2[68].m128i_i8[8] )
        {
          v29 = *(_QWORD *)(v8 + 8LL * (v22 + 1));
          if ( ((a2[46].m128i_i64[1] - a2[46].m128i_i64[0]) >> 3) + ((a2[45].m128i_i64[0] - a2[44].m128i_i64[1]) >> 4) > v29
            && (a2->m128i_i32[2] <= (unsigned int)v29 || !*(_QWORD *)(a2->m128i_i64[0] + 8LL * (unsigned int)v29)) )
          {
            v39 = v29;
            sub_A0FFA0(a2, v29, (__int64)&v59, v29);
            sub_A10370(a2, (__int64)&v59, v33, v34, v35);
            LODWORD(v29) = v39;
          }
          v30 = (_BYTE *)sub_A07560((__int64)a2, v29);
          if ( *v30 == 2 )
          {
LABEL_62:
            v4 = v40;
            v7 = v42;
            if ( (v50 & 2) != 0 )
              goto LABEL_60;
            if ( (v50 & 1) == 0 )
              goto LABEL_3;
LABEL_23:
            if ( v49 )
              (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v49 + 8LL))(v49);
            goto LABEL_3;
          }
          if ( (unsigned __int8)(*v30 - 5) > 0x1Fu )
          {
            v58 = 1;
            v7 = v42;
            v31 = "Invalid metadata attachment";
            goto LABEL_43;
          }
          v32 = (unsigned int)v27[1];
          if ( a2[68].m128i_i8[9] && (_DWORD)v32 == 18 )
          {
            v38 = sub_A87430(v30);
            v32 = (unsigned int)v27[1];
            v30 = (_BYTE *)v38;
          }
          if ( (_DWORD)v32 == 1 )
          {
            v36 = sub_A849A0(v30);
            v32 = (unsigned int)v27[1];
            v30 = (_BYTE *)v36;
          }
          sub_B99FD0(v41, v32, v30);
        }
        v22 += 2;
        if ( v22 == v43 )
          goto LABEL_62;
        v8 = v69;
      }
    }
    sub_A119B0(v7, a2, a3, v69, v70);
    if ( (v53 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *a1 = v53 & 0xFFFFFFFFFFFFFFFELL | 1;
      goto LABEL_44;
    }
    if ( (v50 & 2) != 0 )
      goto LABEL_60;
    if ( (v50 & 1) != 0 )
      goto LABEL_23;
  }
  v58 = 1;
  v31 = "Invalid record";
LABEL_43:
  v53 = (unsigned __int64)v31;
  v57 = 3;
  sub_A01DB0(a1, (__int64)v7);
LABEL_44:
  if ( (v50 & 2) != 0 )
LABEL_60:
    sub_9CE230(&v49);
  if ( (v50 & 1) != 0 && v49 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v49 + 8LL))(v49);
LABEL_7:
  v13 = (__int64 *)&v49;
  v49 = v65;
  v50 = v66;
  v51 = v67;
  v52 = v68;
  v53 = (unsigned __int64)v61;
  v54 = v62;
  v55 = v63;
  v56 = v64;
  sub_A01C60(v7, (__int64 *)&v49);
  v14 = v59;
  if ( v59 )
  {
    v15 = (__int64 *)v64;
    v16 = v68 + 8;
    if ( v68 + 8 > v64 )
    {
      do
      {
        v17 = *v15++;
        j_j___libc_free_0(v17, 512);
      }
      while ( v16 > (unsigned __int64)v15 );
      v14 = v59;
    }
    v13 = (__int64 *)(8 * v60);
    j_j___libc_free_0(v14, 8 * v60);
  }
  if ( (_BYTE *)v69 != v71 )
    _libc_free(v69, v13);
  return a1;
}
