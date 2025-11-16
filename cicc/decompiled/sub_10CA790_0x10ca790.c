// Function: sub_10CA790
// Address: 0x10ca790
//
__int64 __fastcall sub_10CA790(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned __int8 **v6; // rdx
  unsigned __int8 *v7; // r13
  unsigned __int8 *v8; // r15
  unsigned __int32 v9; // r9d
  __int64 v10; // rsi
  int v11; // edx
  _BYTE *v12; // rdi
  __int64 v14; // rax
  unsigned __int8 *v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // r11
  __int64 v23; // r10
  unsigned __int32 v24; // ecx
  char v25; // bl
  __int64 v26; // rax
  __int8 *v27; // rcx
  __int64 v28; // r9
  __int64 v29; // rax
  __m128i *v30; // rax
  unsigned __int64 v31; // rdx
  bool v32; // zf
  __int32 v33; // eax
  __int32 v34; // eax
  unsigned int v35; // r13d
  void *v36; // rax
  _QWORD *v37; // rdi
  size_t v38; // rdx
  int v39; // eax
  __int64 v40; // rax
  unsigned int v41; // eax
  _BYTE *v42; // rcx
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  __int8 *v45; // r12
  _BYTE *v46; // rbx
  _BYTE *v47; // rax
  __int32 v48; // eax
  __int64 v49; // rax
  __m128i *v50; // rax
  int v51; // eax
  __int64 *v52; // [rsp+8h] [rbp-118h]
  __int64 v53; // [rsp+10h] [rbp-110h]
  unsigned int v54; // [rsp+20h] [rbp-100h]
  unsigned int v55; // [rsp+20h] [rbp-100h]
  unsigned __int64 v56; // [rsp+28h] [rbp-F8h]
  _BYTE *v57; // [rsp+30h] [rbp-F0h]
  __int64 v58; // [rsp+38h] [rbp-E8h]
  unsigned __int32 v59; // [rsp+38h] [rbp-E8h]
  int v60; // [rsp+38h] [rbp-E8h]
  __int64 v61; // [rsp+40h] [rbp-E0h]
  __int64 v62; // [rsp+40h] [rbp-E0h]
  __int64 v63; // [rsp+40h] [rbp-E0h]
  unsigned int v64; // [rsp+40h] [rbp-E0h]
  _BYTE *v65; // [rsp+40h] [rbp-E0h]
  bool v66; // [rsp+40h] [rbp-E0h]
  unsigned __int32 v67; // [rsp+48h] [rbp-D8h]
  unsigned __int32 v68; // [rsp+48h] [rbp-D8h]
  __int64 v69; // [rsp+48h] [rbp-D8h]
  unsigned __int64 **v70; // [rsp+48h] [rbp-D8h]
  _BYTE *v71; // [rsp+58h] [rbp-C8h] BYREF
  __m128i v72; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v73; // [rsp+70h] [rbp-B0h] BYREF
  unsigned __int64 v74; // [rsp+80h] [rbp-A0h]
  _BYTE *v75; // [rsp+90h] [rbp-90h] BYREF
  __int64 v76; // [rsp+98h] [rbp-88h]
  _BYTE v77[32]; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v78; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v79; // [rsp+D0h] [rbp-50h]
  _QWORD src[9]; // [rsp+D8h] [rbp-48h] BYREF

  v4 = a1;
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
  {
    v6 = *(unsigned __int8 ***)(a3 - 8);
    v7 = *v6;
    if ( **v6 <= 0x1Cu )
    {
LABEL_17:
      *(_BYTE *)(a1 + 48) = 0;
      return v4;
    }
  }
  else
  {
    v6 = (unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    v7 = *v6;
    if ( **v6 <= 0x1Cu )
      goto LABEL_17;
  }
  v8 = v6[4];
  if ( *v8 <= 0x1Cu )
    goto LABEL_17;
  v9 = sub_BCB060(*(_QWORD *)(a3 + 8));
  v75 = v77;
  v76 = 0x300000000LL;
  v10 = *v7;
  if ( (unsigned int)(v10 - 42) > 0x11 )
  {
    LOBYTE(v11) = *v8;
LABEL_6:
    if ( *v7 == 68 )
    {
      if ( (_BYTE)v11 != 68 )
      {
        v15 = v7;
        v7 = v8;
        v8 = v15;
      }
    }
    else if ( (_BYTE)v11 != 68 )
    {
LABEL_8:
      v12 = v77;
LABEL_9:
      *(_BYTE *)(v4 + 48) = 0;
      goto LABEL_10;
    }
    LOBYTE(v79) = 0;
    v78.m128i_i64[0] = (__int64)&v71;
    v78.m128i_i64[1] = (__int64)&v72;
    v16 = *((_QWORD *)v7 + 2);
    if ( !v16 )
      goto LABEL_8;
    if ( *(_QWORD *)(v16 + 8) )
      goto LABEL_8;
    if ( *v7 != 54 )
      goto LABEL_8;
    v67 = v9;
    if ( !*((_QWORD *)v7 - 8) )
      goto LABEL_8;
    v71 = (_BYTE *)*((_QWORD *)v7 - 8);
    v10 = *((_QWORD *)v7 - 4);
    if ( !(unsigned __int8)sub_991580((__int64)&v78.m128i_i64[1], v10) )
    {
LABEL_25:
      v12 = v75;
      goto LABEL_9;
    }
    if ( *v8 != 68 )
      goto LABEL_29;
    if ( !*((_QWORD *)v8 - 4) )
      goto LABEL_29;
    if ( *v71 != 68 )
      goto LABEL_29;
    v40 = *((_QWORD *)v71 - 4);
    v63 = *((_QWORD *)v8 - 4);
    v57 = v71;
    if ( !v40 )
      goto LABEL_29;
    v60 = sub_BCB060(*(_QWORD *)(v40 + 8));
    v41 = sub_BCB060(*(_QWORD *)(v63 + 8));
    v42 = v57;
    v28 = v67;
    v43 = v41;
    v64 = *(_DWORD *)(v72.m128i_i64[0] + 8);
    if ( v64 > 0x40 )
    {
      v55 = v67;
      v56 = v41;
      v70 = (unsigned __int64 **)v72.m128i_i64[0];
      v51 = sub_C444A0(v72.m128i_i64[0]);
      v42 = v57;
      v28 = v55;
      if ( v64 - v51 > 0x40 )
        goto LABEL_29;
      v44 = **v70;
      if ( v56 > v44 )
        goto LABEL_29;
    }
    else
    {
      v44 = *(_QWORD *)v72.m128i_i64[0];
      if ( v43 > *(_QWORD *)v72.m128i_i64[0] )
        goto LABEL_29;
    }
    v31 = (unsigned int)(v28 - v60);
    if ( v31 >= v44 )
    {
      v27 = (__int8 *)*((_QWORD *)v42 + 2);
      if ( v27 )
      {
        v45 = v27;
        v54 = v28;
        v53 = a3;
        while ( 1 )
        {
          v46 = (_BYTE *)*((_QWORD *)v45 + 3);
          if ( *v46 == 58 )
          {
            v31 = *((_QWORD *)v46 - 8);
            if ( v31 )
            {
              v47 = (_BYTE *)*((_QWORD *)v46 - 4);
              if ( v47 )
              {
                if ( *v47 == 68 )
                {
                  v10 = *((_QWORD *)v46 - 4);
                  v47 = (_BYTE *)*((_QWORD *)v46 - 8);
                  v31 = v10;
                }
                v27 = &v72.m128i_i8[8];
                v78.m128i_i64[0] = (__int64)v8;
                LOBYTE(v79) = 0;
                v78.m128i_i64[1] = (__int64)&v72.m128i_i64[1];
                if ( *v47 == 54 )
                {
                  v10 = *((_QWORD *)v47 - 8);
                  v65 = (_BYTE *)v31;
                  if ( (unsigned __int8 *)v10 == v8 )
                  {
                    if ( v10 )
                    {
                      v10 = *((_QWORD *)v47 - 4);
                      if ( (unsigned __int8)sub_991580((__int64)&v78.m128i_i64[1], v10) )
                      {
                        v31 = (unsigned __int64)v65;
                        if ( v71 == v65 )
                        {
                          v10 = (__int64)v46;
                          if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a2 + 80), (__int64)v46, v53) )
                          {
                            v52 = (__int64 *)v72.m128i_i64[0];
                            sub_9865C0((__int64)&v73, v72.m128i_i64[1]);
                            sub_C45EE0((__int64)&v73, v52);
                            v48 = v73.m128i_i32[2];
                            v10 = v54;
                            v73.m128i_i32[2] = 0;
                            v78.m128i_i32[2] = v48;
                            v78.m128i_i64[0] = v73.m128i_i64[0];
                            v66 = sub_D94970((__int64)&v78, (_QWORD *)v54);
                            sub_969240(v78.m128i_i64);
                            sub_969240(v73.m128i_i64);
                            if ( v66 )
                              break;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          v45 = (__int8 *)*((_QWORD *)v45 + 1);
          if ( !v45 )
          {
            v4 = a1;
            goto LABEL_83;
          }
        }
        v78.m128i_i64[0] = (__int64)v46;
        v78.m128i_i64[1] = (__int64)v46;
        v10 = v72.m128i_i64[0];
        v4 = a1;
        v49 = sub_AD8D80(*((_QWORD *)v7 + 1), v72.m128i_i64[0]);
        LODWORD(v76) = 0;
        v79 = v49;
        if ( HIDWORD(v76) <= 2 )
        {
          v10 = (__int64)v77;
          sub_C8D5F0((__int64)&v75, v77, 3u, 8u, (__int64)v77, v28);
        }
        v50 = (__m128i *)&v75[8 * (unsigned int)v76];
        *v50 = _mm_loadu_si128(&v78);
        v31 = v79;
        LODWORD(v76) = v76 + 3;
        v50[1].m128i_i64[0] = v79;
      }
LABEL_83:
      if ( !(_DWORD)v76 )
        goto LABEL_25;
LABEL_84:
      v33 = 180;
      goto LABEL_47;
    }
LABEL_29:
    *(_BYTE *)(a1 + 48) = 0;
    v12 = v75;
    goto LABEL_10;
  }
  v11 = *v8;
  if ( (unsigned int)(v11 - 42) > 0x11 )
    goto LABEL_6;
  v14 = *((_QWORD *)v7 + 2);
  if ( !v14 )
    goto LABEL_15;
  if ( *(_QWORD *)(v14 + 8) )
    goto LABEL_15;
  v10 = (unsigned int)(v10 - 54);
  v68 = v9;
  if ( (unsigned int)v10 > 1 )
    goto LABEL_15;
  v17 = (__int64 *)sub_986520((__int64)v7);
  v18 = *v17;
  if ( !*v17 )
    goto LABEL_15;
  v72.m128i_i64[0] = *v17;
  v61 = v18;
  v19 = *(_QWORD *)(sub_986520((__int64)v7) + 32);
  if ( !v19
    || (v20 = *((_QWORD *)v8 + 2)) == 0
    || *(_QWORD *)(v20 + 8)
    || (v58 = v19, (unsigned int)*v8 - 54 > 1)
    || (v21 = (__int64 *)sub_986520((__int64)v8), (v10 = *v21) == 0)
    || (v72.m128i_i64[1] = *v21, (v22 = *(_QWORD *)(sub_986520((__int64)v8) + 32)) == 0)
    || *v8 == *v7 )
  {
LABEL_15:
    *(_BYTE *)(a1 + 48) = 0;
    v12 = v77;
    goto LABEL_10;
  }
  v23 = v58;
  if ( *v7 == 55 )
  {
    v72.m128i_i64[0] = v10;
    v23 = v22;
    v72.m128i_i64[1] = v61;
    v22 = v58;
  }
  v24 = v68;
  v10 = v23;
  v79 = (__int64)&v72;
  v78.m128i_i64[0] = a3;
  v25 = 1;
  v59 = v68;
  v62 = v22;
  v69 = v23;
  v78.m128i_i64[1] = a2;
  src[0] = &v72.m128i_i64[1];
  v26 = sub_10C9CD0((__int64)&v78, v23, v22, v24);
  v28 = v59;
  if ( !v26 )
  {
    v10 = v62;
    v26 = sub_10C9CD0((__int64)&v78, v62, v69, v59);
    if ( !v26 )
      goto LABEL_29;
    v25 = 0;
  }
  v74 = v26;
  v29 = 0;
  LODWORD(v76) = 0;
  v73 = v72;
  if ( HIDWORD(v76) <= 2 )
  {
    v10 = (__int64)v77;
    sub_C8D5F0((__int64)&v75, v77, 3u, 8u, (__int64)v77, v28);
    v29 = 8LL * (unsigned int)v76;
  }
  v30 = (__m128i *)&v75[v29];
  *v30 = _mm_loadu_si128(&v73);
  v31 = v74;
  v32 = (_DWORD)v76 == -3;
  LODWORD(v76) = v76 + 3;
  v30[1].m128i_i64[0] = v74;
  if ( v32 )
    goto LABEL_25;
  v33 = 181;
  if ( v25 )
    goto LABEL_84;
LABEL_47:
  v78.m128i_i32[0] = v33;
  v10 = (__int64)&v75;
  v79 = 0x300000000LL;
  v78.m128i_i64[1] = (__int64)src;
  sub_10B8390((__int64)&v78.m128i_i64[1], (__int64)&v75, v31, (__int64)v27, (__int64)v77, v28);
  v34 = v78.m128i_i32[0];
  *(_QWORD *)(v4 + 16) = 0x300000000LL;
  v35 = v79;
  *(_DWORD *)v4 = v34;
  v36 = (void *)(v4 + 24);
  *(_QWORD *)(v4 + 8) = v4 + 24;
  if ( !v35 )
  {
    v37 = (_QWORD *)v78.m128i_i64[1];
LABEL_49:
    *(_BYTE *)(v4 + 48) = 1;
    if ( v37 != src )
      _libc_free(v37, v10);
    goto LABEL_51;
  }
  if ( (_QWORD *)v78.m128i_i64[1] == src )
  {
    v37 = src;
    v38 = 8LL * v35;
    if ( v35 <= 3
      || (v10 = v4 + 24,
          sub_C8D5F0(v4 + 8, (const void *)(v4 + 24), v35, 8u, (__int64)v77, v35),
          v37 = (_QWORD *)v78.m128i_i64[1],
          v36 = *(void **)(v4 + 8),
          (v38 = 8LL * (unsigned int)v79) != 0) )
    {
      v10 = (__int64)v37;
      memcpy(v36, v37, v38);
      v37 = (_QWORD *)v78.m128i_i64[1];
    }
    *(_DWORD *)(v4 + 16) = v35;
    goto LABEL_49;
  }
  v39 = HIDWORD(v79);
  *(_QWORD *)(v4 + 8) = v78.m128i_i64[1];
  *(_DWORD *)(v4 + 16) = v35;
  *(_DWORD *)(v4 + 20) = v39;
  *(_BYTE *)(v4 + 48) = 1;
LABEL_51:
  v12 = v75;
LABEL_10:
  if ( v12 != v77 )
    _libc_free(v12, v10);
  return v4;
}
