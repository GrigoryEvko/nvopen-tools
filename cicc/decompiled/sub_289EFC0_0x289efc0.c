// Function: sub_289EFC0
// Address: 0x289efc0
//
void __fastcall sub_289EFC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // edx
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 **v11; // rax
  __int64 v12; // rax
  _BYTE **v13; // rbx
  _BYTE *v14; // r15
  __int64 v15; // rax
  _BYTE *v16; // r13
  __int64 v17; // rax
  unsigned __int8 *v18; // r14
  __int64 (__fastcall *v19)(__int64, _BYTE *, unsigned __int8 *); // rax
  _BYTE *v20; // r12
  __int64 v21; // rax
  unsigned __int8 *v22; // r13
  __int64 (__fastcall *v23)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v24; // rax
  __int64 v25; // r14
  unsigned int *v26; // r13
  unsigned int *v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  _QWORD *v30; // rax
  unsigned int *v31; // r14
  unsigned int *v32; // r13
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v37; // eax
  __int64 v38; // rdx
  __m128i v39; // xmm0
  int v41; // [rsp+44h] [rbp-35Ch]
  __int64 v42; // [rsp+48h] [rbp-358h]
  __int64 v43; // [rsp+50h] [rbp-350h]
  _BYTE *v44; // [rsp+78h] [rbp-328h]
  __int64 v45; // [rsp+80h] [rbp-320h]
  __int64 v46; // [rsp+88h] [rbp-318h]
  unsigned int v47; // [rsp+94h] [rbp-30Ch] BYREF
  unsigned int v48; // [rsp+98h] [rbp-308h]
  _BYTE v49[32]; // [rsp+A0h] [rbp-300h] BYREF
  __int16 v50; // [rsp+C0h] [rbp-2E0h]
  unsigned int *v51; // [rsp+D0h] [rbp-2D0h] BYREF
  __int64 v52; // [rsp+D8h] [rbp-2C8h]
  _BYTE v53[32]; // [rsp+E0h] [rbp-2C0h] BYREF
  __int64 v54; // [rsp+100h] [rbp-2A0h]
  __int64 v55; // [rsp+108h] [rbp-298h]
  __int64 v56; // [rsp+110h] [rbp-290h]
  _QWORD *v57; // [rsp+118h] [rbp-288h]
  void **v58; // [rsp+120h] [rbp-280h]
  void **v59; // [rsp+128h] [rbp-278h]
  __int64 v60; // [rsp+130h] [rbp-270h]
  int v61; // [rsp+138h] [rbp-268h]
  __int16 v62; // [rsp+13Ch] [rbp-264h]
  char v63; // [rsp+13Eh] [rbp-262h]
  __int64 v64; // [rsp+140h] [rbp-260h]
  __int64 v65; // [rsp+148h] [rbp-258h]
  void *v66; // [rsp+150h] [rbp-250h] BYREF
  void *v67; // [rsp+158h] [rbp-248h] BYREF
  _BYTE *v68; // [rsp+160h] [rbp-240h] BYREF
  __int64 v69; // [rsp+168h] [rbp-238h]
  _BYTE v70[128]; // [rsp+170h] [rbp-230h] BYREF
  __m128i v71; // [rsp+1F0h] [rbp-1B0h] BYREF
  bool v72; // [rsp+200h] [rbp-1A0h]
  _BYTE *v73; // [rsp+210h] [rbp-190h] BYREF
  int v74; // [rsp+218h] [rbp-188h]
  _BYTE v75[160]; // [rsp+220h] [rbp-180h] BYREF
  __m128i v76; // [rsp+2C0h] [rbp-E0h] BYREF
  _BYTE v77[16]; // [rsp+2D0h] [rbp-D0h] BYREF
  __int16 v78; // [rsp+2E0h] [rbp-C0h]
  __m128i v79; // [rsp+350h] [rbp-50h]
  bool v80; // [rsp+360h] [rbp-40h]

  v68 = v70;
  v72 = dword_5003CC8 == 0;
  v69 = 0x1000000000LL;
  v71 = 0u;
  v57 = (_QWORD *)sub_BD5C60(a2);
  v58 = &v66;
  v59 = &v67;
  v51 = (unsigned int *)v53;
  v66 = &unk_49DA100;
  v52 = 0x200000000LL;
  v62 = 512;
  LOWORD(v56) = 0;
  v67 = &unk_49DA0B0;
  v60 = 0;
  v61 = 0;
  v63 = 7;
  v64 = 0;
  v65 = 0;
  v54 = 0;
  v55 = 0;
  sub_D5F1F0((__int64)&v51, a2);
  v2 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v43 = *(_QWORD *)(*(_QWORD *)(a2 - 32 * v2) + 8LL);
  sub_28940A0((__int64)&v47, *(_QWORD *)(a2 + 32 * (1 - v2)), *(_QWORD *)(a2 + 32 * (2 - v2)));
  sub_2895860((__int64)&v73, a1, v5, v3, v4);
  v8 = v47;
  v9 = v48;
  if ( v75[144] )
  {
    v10 = v47;
    v41 = v48;
    if ( v47 )
    {
LABEL_3:
      v42 = v10;
      v45 = 0;
      while ( 1 )
      {
        v11 = (__int64 **)sub_BCDA70(*(__int64 **)(v43 + 24), v41);
        v12 = sub_ACADE0(v11);
        v13 = (_BYTE **)v73;
        v14 = (_BYTE *)v12;
        v44 = &v73[8 * v74];
        if ( v73 != v44 )
          break;
LABEL_31:
        v35 = (unsigned int)v69;
        v36 = (unsigned int)v69 + 1LL;
        if ( v36 > HIDWORD(v69) )
        {
          sub_C8D5F0((__int64)&v68, v70, v36, 8u, v6, v7);
          v35 = (unsigned int)v69;
        }
        ++v45;
        *(_QWORD *)&v68[8 * v35] = v14;
        v37 = v69 + 1;
        LODWORD(v69) = v69 + 1;
        if ( v42 == v45 )
        {
          v8 = v47;
          v9 = v48;
          goto LABEL_35;
        }
      }
      v46 = 0;
      while ( 1 )
      {
        v50 = 257;
        v16 = *v13;
        v17 = sub_BCB2E0(v57);
        v18 = (unsigned __int8 *)sub_ACD640(v17, v45, 0);
        v19 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v58 + 12);
        if ( v19 != sub_948070 )
          break;
        if ( *v16 <= 0x15u && *v18 <= 0x15u )
        {
          v20 = (_BYTE *)sub_AD5840((__int64)v16, v18, 0);
          goto LABEL_16;
        }
LABEL_24:
        v78 = 257;
        v30 = sub_BD2C40(72, 2u);
        v20 = v30;
        if ( v30 )
          sub_B4DE80((__int64)v30, (__int64)v16, (__int64)v18, (__int64)&v76, 0, 0);
        (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v59 + 2))(v59, v20, v49, v55, v56);
        v31 = v51;
        v32 = &v51[4 * (unsigned int)v52];
        if ( v51 != v32 )
        {
          do
          {
            v33 = *((_QWORD *)v31 + 1);
            v34 = *v31;
            v31 += 4;
            sub_B99FD0((__int64)v20, v34, v33);
          }
          while ( v32 != v31 );
        }
LABEL_17:
        v50 = 257;
        v21 = sub_BCB2E0(v57);
        v22 = (unsigned __int8 *)sub_ACD640(v21, v46, 0);
        v23 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))*((_QWORD *)*v58 + 13);
        if ( v23 == sub_948040 )
        {
          if ( *v14 > 0x15u || *v20 > 0x15u || *v22 > 0x15u )
          {
LABEL_19:
            v78 = 257;
            v24 = sub_BD2C40(72, 3u);
            v25 = (__int64)v24;
            if ( v24 )
              sub_B4DFA0((__int64)v24, (__int64)v14, (__int64)v20, (__int64)v22, (__int64)&v76, 0, 0, 0);
            (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v59 + 2))(v59, v25, v49, v55, v56);
            v26 = v51;
            v27 = &v51[4 * (unsigned int)v52];
            if ( v51 != v27 )
            {
              do
              {
                v28 = *((_QWORD *)v26 + 1);
                v29 = *v26;
                v26 += 4;
                sub_B99FD0(v25, v29, v28);
              }
              while ( v27 != v26 );
            }
            v14 = (_BYTE *)v25;
            goto LABEL_11;
          }
          v15 = sub_AD5A90((__int64)v14, v20, v22, 0);
        }
        else
        {
          v15 = v23((__int64)v58, v14, v20, v22);
        }
        if ( !v15 )
          goto LABEL_19;
        v14 = (_BYTE *)v15;
LABEL_11:
        ++v46;
        if ( v44 == (_BYTE *)++v13 )
          goto LABEL_31;
      }
      v20 = (_BYTE *)v19((__int64)v58, v16, v18);
LABEL_16:
      if ( v20 )
        goto LABEL_17;
      goto LABEL_24;
    }
  }
  else
  {
    v10 = v48;
    v41 = v47;
    if ( v48 )
      goto LABEL_3;
  }
  v37 = v69;
LABEL_35:
  ++v71.m128i_i32[3];
  v76.m128i_i64[0] = (__int64)v77;
  v76.m128i_i64[1] = 0x1000000000LL;
  v38 = 2 * v9 * v8;
  v71.m128i_i32[2] += v38;
  if ( v37 )
    sub_2894AD0((__int64)&v76, (__int64)&v68, v38, 0x1000000000LL, v6, v7);
  v39 = _mm_loadu_si128(&v71);
  v80 = v72;
  v79 = v39;
  sub_289E450(a1, a2, &v76, &v51, v6, v7);
  if ( (_BYTE *)v76.m128i_i64[0] != v77 )
    _libc_free(v76.m128i_u64[0]);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  nullsub_61();
  v66 = &unk_49DA100;
  nullsub_63();
  if ( v51 != (unsigned int *)v53 )
    _libc_free((unsigned __int64)v51);
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
}
