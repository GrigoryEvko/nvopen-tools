// Function: sub_2941410
// Address: 0x2941410
//
__int64 __fastcall sub_2941410(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned __int64 v3; // r12
  __int64 v5; // r8
  __int64 v6; // r9
  _BYTE *v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int32 v10; // r14d
  _QWORD *v11; // rax
  _BYTE *v12; // rdx
  _QWORD *i; // rdx
  _QWORD *v14; // rax
  unsigned int v15; // ebx
  unsigned __int32 v16; // esi
  __int32 v17; // eax
  unsigned __int32 v18; // r13d
  unsigned __int32 v19; // r15d
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int8 *v23; // r15
  __int64 (__fastcall *v24)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rbx
  unsigned int *v28; // r12
  unsigned int *v29; // r15
  __int64 v30; // rdx
  unsigned int v31; // esi
  unsigned int v32; // ebx
  __m128i v33; // rax
  char v34; // al
  _QWORD *v35; // rdx
  __int64 v36; // r12
  _BYTE *v37; // rax
  __int64 v38; // rax
  __m128i v39; // rax
  char v40; // al
  _QWORD *v41; // rdx
  __int64 v42; // rax
  __m128i v43; // xmm3
  __m128i v44; // xmm1
  __int64 v45; // [rsp+10h] [rbp-2D0h]
  __int64 v46; // [rsp+10h] [rbp-2D0h]
  __int64 v47; // [rsp+20h] [rbp-2C0h]
  unsigned __int64 v48; // [rsp+28h] [rbp-2B8h]
  __int64 v49; // [rsp+40h] [rbp-2A0h]
  unsigned int v50; // [rsp+48h] [rbp-298h]
  __int64 v51; // [rsp+48h] [rbp-298h]
  _BYTE *v52; // [rsp+58h] [rbp-288h]
  _QWORD *v53; // [rsp+58h] [rbp-288h]
  _QWORD *v54; // [rsp+58h] [rbp-288h]
  unsigned int v55; // [rsp+58h] [rbp-288h]
  _BYTE *v56; // [rsp+60h] [rbp-280h]
  unsigned __int8 v57; // [rsp+6Fh] [rbp-271h]
  __m128i v58; // [rsp+70h] [rbp-270h] BYREF
  __int64 v59; // [rsp+88h] [rbp-258h]
  unsigned __int8 v60; // [rsp+90h] [rbp-250h]
  __m128i v61; // [rsp+A0h] [rbp-240h] BYREF
  __m128i v62; // [rsp+B0h] [rbp-230h] BYREF
  __int64 v63; // [rsp+C0h] [rbp-220h]
  _QWORD *v64; // [rsp+D0h] [rbp-210h] BYREF
  __int64 v65; // [rsp+D8h] [rbp-208h]
  __int16 v66; // [rsp+F0h] [rbp-1F0h]
  __m128i v67; // [rsp+100h] [rbp-1E0h] BYREF
  __m128i v68; // [rsp+110h] [rbp-1D0h]
  __int64 v69; // [rsp+120h] [rbp-1C0h]
  _BYTE *v70; // [rsp+130h] [rbp-1B0h] BYREF
  __int64 v71; // [rsp+138h] [rbp-1A8h]
  _BYTE v72[64]; // [rsp+140h] [rbp-1A0h] BYREF
  unsigned int *v73; // [rsp+180h] [rbp-160h] BYREF
  int v74; // [rsp+188h] [rbp-158h]
  char v75; // [rsp+190h] [rbp-150h] BYREF
  __int64 v76; // [rsp+1B8h] [rbp-128h]
  __int64 v77; // [rsp+1C0h] [rbp-120h]
  _QWORD *v78; // [rsp+1C8h] [rbp-118h]
  __int64 v79; // [rsp+1D0h] [rbp-110h]
  __int64 v80; // [rsp+1D8h] [rbp-108h]
  void *v81; // [rsp+200h] [rbp-E0h]
  __m128i v82[5]; // [rsp+210h] [rbp-D0h] BYREF
  char *v83; // [rsp+260h] [rbp-80h]
  char v84; // [rsp+270h] [rbp-70h] BYREF

  v2 = a1;
  v3 = a2;
  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, (unsigned __int8 *)a2) )
    return 0;
  sub_2939E80((__int64)&v58, a1, *(_QWORD *)(a2 + 8));
  v57 = v60;
  if ( !v60 )
    return 0;
  sub_23D0AB0((__int64)&v73, a2, 0, 0, 0);
  sub_293CE40(v82, (_QWORD *)a1, a2, *(_QWORD *)(a2 - 96), &v58);
  v7 = *(_BYTE **)(a2 - 64);
  v8 = v58.m128i_u32[3];
  v71 = 0x800000000LL;
  v9 = *(_QWORD *)(a2 - 32);
  v56 = v7;
  v10 = v58.m128i_i32[3];
  v11 = v72;
  v12 = v72;
  v70 = v72;
  if ( v58.m128i_i32[3] )
  {
    if ( v58.m128i_u32[3] > 8uLL )
    {
      sub_C8D5F0((__int64)&v70, v72, v58.m128i_u32[3], 8u, v5, v6);
      v12 = v70;
      v11 = &v70[8 * (unsigned int)v71];
    }
    for ( i = &v12[8 * v8]; i != v11; ++v11 )
    {
      if ( v11 )
        *v11 = 0;
    }
    LODWORD(v71) = v10;
  }
  if ( *(_BYTE *)v9 == 17 )
  {
    v14 = *(_QWORD **)(v9 + 24);
    if ( *(_DWORD *)(v9 + 32) > 0x40u )
      v14 = (_QWORD *)*v14;
    v50 = (unsigned int)v14;
    v15 = 0;
    v16 = (unsigned int)v14 / v58.m128i_i32[2];
    v17 = v58.m128i_i32[3];
    if ( v58.m128i_i32[3] )
    {
      v18 = v16;
      v48 = v3;
      while ( 1 )
      {
        v20 = v15;
        if ( v18 != v15 )
        {
          v21 = sub_293BC00((__int64)v82, v15);
          goto LABEL_23;
        }
        v19 = v58.m128i_u32[2];
        if ( v17 - 1 == v18 && v59 )
        {
          if ( (unsigned int)*(unsigned __int8 *)(v59 + 8) - 17 <= 1 && v58.m128i_i32[2] > 1u )
            goto LABEL_35;
LABEL_20:
          ++v15;
          *(_QWORD *)&v70[8 * v20] = v56;
          v17 = v58.m128i_i32[3];
          if ( v58.m128i_i32[3] <= v15 )
            goto LABEL_24;
        }
        else
        {
          if ( v58.m128i_i32[2] <= 1u )
            goto LABEL_20;
LABEL_35:
          v66 = 257;
          v52 = (_BYTE *)sub_293BC00((__int64)v82, v15);
          v22 = sub_BCB2E0(v78);
          v23 = (unsigned __int8 *)sub_ACD640(v22, v50 % v19, 0);
          v24 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v79 + 104LL);
          if ( v24 == sub_948040 )
          {
            if ( *v52 > 0x15u || *v56 > 0x15u || *v23 > 0x15u )
            {
LABEL_41:
              LOWORD(v69) = 257;
              v25 = sub_BD2C40(72, 3u);
              if ( v25 )
              {
                v26 = (__int64)v52;
                v53 = v25;
                sub_B4DFA0((__int64)v25, v26, (__int64)v56, (__int64)v23, (__int64)&v67, 0, 0, 0);
                v25 = v53;
              }
              v54 = v25;
              (*(void (__fastcall **)(__int64, _QWORD *, _QWORD **, __int64, __int64))(*(_QWORD *)v80 + 16LL))(
                v80,
                v25,
                &v64,
                v76,
                v77);
              v21 = (__int64)v54;
              if ( v73 != &v73[4 * v74] )
              {
                v55 = v15;
                v27 = v21;
                v46 = v20;
                v28 = v73;
                v29 = &v73[4 * v74];
                do
                {
                  v30 = *((_QWORD *)v28 + 1);
                  v31 = *v28;
                  v28 += 4;
                  sub_B99FD0(v27, v31, v30);
                }
                while ( v29 != v28 );
                v21 = v27;
                v20 = v46;
                v15 = v55;
              }
              goto LABEL_23;
            }
            v21 = sub_AD5A90((__int64)v52, v56, v23, 0);
          }
          else
          {
            v21 = v24(v79, v52, v56, v23);
          }
          if ( !v21 )
            goto LABEL_41;
LABEL_23:
          ++v15;
          *(_QWORD *)&v70[8 * v20] = v21;
          v17 = v58.m128i_i32[3];
          if ( v58.m128i_i32[3] <= v15 )
          {
LABEL_24:
            v2 = a1;
            v3 = v48;
            break;
          }
        }
      }
    }
  }
  else
  {
    if ( !*(_BYTE *)(a1 + 1128) || v58.m128i_i32[2] > 1u )
    {
      v57 = 0;
      goto LABEL_26;
    }
    if ( v58.m128i_i32[3] )
    {
      v32 = 0;
      do
      {
        LODWORD(v64) = v32;
        v66 = 265;
        v33.m128i_i64[0] = (__int64)sub_BD5D20(v9);
        v61 = v33;
        v62.m128i_i64[0] = (__int64)".is.";
        v34 = v66;
        LOWORD(v63) = 773;
        if ( (_BYTE)v66 )
        {
          if ( (_BYTE)v66 == 1 )
          {
            v44 = _mm_loadu_si128(&v62);
            v67 = _mm_loadu_si128(&v61);
            v69 = v63;
            v68 = v44;
          }
          else
          {
            if ( HIBYTE(v66) == 1 )
            {
              v35 = v64;
              v47 = v65;
            }
            else
            {
              v35 = &v64;
              v34 = 2;
            }
            v68.m128i_i64[0] = (__int64)v35;
            LOBYTE(v69) = 2;
            v67.m128i_i64[0] = (__int64)&v61;
            BYTE1(v69) = v34;
            v68.m128i_i64[1] = v47;
          }
        }
        else
        {
          LOWORD(v69) = 256;
        }
        v36 = v32;
        v37 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v9 + 8), v32, 0);
        v49 = sub_92B530(&v73, 0x20u, v9, v37, (__int64)&v67);
        v38 = sub_293BC00((__int64)v82, v32);
        LODWORD(v64) = v32;
        v51 = v38;
        v66 = 265;
        v39.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v61 = v39;
        v62.m128i_i64[0] = (__int64)".i";
        v40 = v66;
        LOWORD(v63) = 773;
        if ( (_BYTE)v66 )
        {
          if ( (_BYTE)v66 == 1 )
          {
            v43 = _mm_loadu_si128(&v62);
            v67 = _mm_loadu_si128(&v61);
            v69 = v63;
            v68 = v43;
          }
          else
          {
            if ( HIBYTE(v66) == 1 )
            {
              v41 = v64;
              v45 = v65;
            }
            else
            {
              v41 = &v64;
              v40 = 2;
            }
            v68.m128i_i64[0] = (__int64)v41;
            LOBYTE(v69) = 2;
            v67.m128i_i64[0] = (__int64)&v61;
            BYTE1(v69) = v40;
            v68.m128i_i64[1] = v45;
          }
        }
        else
        {
          LOWORD(v69) = 256;
        }
        ++v32;
        v42 = sub_B36550(&v73, v49, (__int64)v56, v51, (__int64)&v67, 0);
        *(_QWORD *)&v70[8 * v36] = v42;
      }
      while ( v58.m128i_i32[3] > v32 );
      v3 = a2;
    }
  }
  sub_293CAB0(v2, v3, (__int64)&v70, (__int64)&v58);
LABEL_26:
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( v83 != &v84 )
    _libc_free((unsigned __int64)v83);
  nullsub_61();
  v81 = &unk_49DA100;
  nullsub_63();
  if ( v73 != (unsigned int *)&v75 )
    _libc_free((unsigned __int64)v73);
  return v57;
}
