// Function: sub_293D130
// Address: 0x293d130
//
__int64 __fastcall sub_293D130(__int64 a1, __int64 a2, unsigned __int8 **a3)
{
  unsigned __int64 v3; // r14
  __int64 v4; // rdx
  __m128i v5; // xmm2
  __m128i v7; // xmm3
  __m128i v8; // xmm4
  unsigned __int64 *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rbx
  __int32 v14; // r12d
  _BYTE *v15; // rdx
  _QWORD *v16; // rax
  _QWORD *i; // rdx
  unsigned int v18; // ebx
  _QWORD *v19; // rdx
  char v20; // al
  __int64 (__fastcall *v21)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned int v22; // r14d
  __int64 v23; // r15
  __int64 v24; // rdx
  unsigned __int8 *v25; // r12
  __int64 v26; // rax
  unsigned __int8 *v27; // r13
  __m128i v28; // rax
  int v29; // r12d
  char *v30; // r13
  char *v31; // r12
  __int64 v32; // rdx
  unsigned int v33; // esi
  __m128i v34; // xmm6
  unsigned __int8 v35; // [rsp+2Fh] [rbp-3D1h]
  __int64 v36; // [rsp+40h] [rbp-3C0h]
  __m128i v39; // [rsp+60h] [rbp-3A0h] BYREF
  __m128i v40; // [rsp+70h] [rbp-390h] BYREF
  __int64 v41; // [rsp+80h] [rbp-380h]
  __m128i v42; // [rsp+90h] [rbp-370h] BYREF
  __m128i v43; // [rsp+A0h] [rbp-360h] BYREF
  __int64 v44; // [rsp+B0h] [rbp-350h]
  __m128i v45; // [rsp+C0h] [rbp-340h] BYREF
  __m128i v46; // [rsp+D0h] [rbp-330h]
  __int64 v47; // [rsp+E0h] [rbp-320h]
  __m128i v48; // [rsp+F0h] [rbp-310h] BYREF
  __m128i v49; // [rsp+100h] [rbp-300h] BYREF
  __int64 v50; // [rsp+110h] [rbp-2F0h]
  _QWORD v51[4]; // [rsp+120h] [rbp-2E0h] BYREF
  __int16 v52; // [rsp+140h] [rbp-2C0h]
  __m128i v53; // [rsp+150h] [rbp-2B0h] BYREF
  __m128i v54; // [rsp+160h] [rbp-2A0h]
  __int64 v55; // [rsp+170h] [rbp-290h]
  char v56[32]; // [rsp+180h] [rbp-280h] BYREF
  __int16 v57; // [rsp+1A0h] [rbp-260h]
  _BYTE *v58; // [rsp+1B0h] [rbp-250h] BYREF
  __int64 v59; // [rsp+1B8h] [rbp-248h]
  _BYTE v60[64]; // [rsp+1C0h] [rbp-240h] BYREF
  char *v61; // [rsp+200h] [rbp-200h] BYREF
  int v62; // [rsp+208h] [rbp-1F8h]
  char v63; // [rsp+210h] [rbp-1F0h] BYREF
  __int64 v64; // [rsp+238h] [rbp-1C8h]
  __int64 v65; // [rsp+240h] [rbp-1C0h]
  __int64 v66; // [rsp+250h] [rbp-1B0h]
  __int64 v67; // [rsp+258h] [rbp-1A8h]
  __int64 v68; // [rsp+260h] [rbp-1A0h]
  int v69; // [rsp+268h] [rbp-198h]
  void *v70; // [rsp+280h] [rbp-180h]
  __m128i v71[5]; // [rsp+290h] [rbp-170h] BYREF
  char *v72; // [rsp+2E0h] [rbp-120h]
  char v73; // [rsp+2F0h] [rbp-110h] BYREF
  __m128i v74[5]; // [rsp+330h] [rbp-D0h] BYREF
  char *v75; // [rsp+380h] [rbp-80h]
  char v76; // [rsp+390h] [rbp-70h] BYREF

  v3 = a2;
  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, (unsigned __int8 *)a2) )
    return 0;
  sub_2939E80((__int64)&v42, a1, *(_QWORD *)(a2 + 8));
  v35 = v44;
  if ( !(_BYTE)v44 )
    return 0;
  v47 = 0;
  v45 = 0;
  v46 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v4 = *(_QWORD *)(**(_QWORD **)(a2 - 8) + 8LL);
    if ( v4 != *(_QWORD *)(a2 + 8) )
      goto LABEL_5;
LABEL_10:
    v7 = _mm_loadu_si128(&v42);
    v8 = _mm_loadu_si128(&v43);
    v47 = v44;
    v45 = v7;
    v46 = v8;
    goto LABEL_11;
  }
  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
  if ( v4 == *(_QWORD *)(a2 + 8) )
    goto LABEL_10;
LABEL_5:
  sub_2939E80((__int64)&v39, a1, v4);
  v5 = _mm_loadu_si128(&v40);
  v45 = _mm_loadu_si128(&v39);
  v47 = v41;
  v46 = v5;
  if ( !(_BYTE)v41 || v42.m128i_i32[2] != v45.m128i_i32[2] )
    return 0;
LABEL_11:
  sub_23D0AB0((__int64)&v61, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v9 = *(unsigned __int64 **)(a2 - 8);
  else
    v9 = (unsigned __int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  sub_293CE40(v71, (_QWORD *)a1, a2, *v9, &v45);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v10 = *(_QWORD *)(a2 - 8);
  else
    v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  sub_293CE40(v74, (_QWORD *)a1, a2, *(_QWORD *)(v10 + 32), &v45);
  v13 = v42.m128i_u32[3];
  v14 = v42.m128i_i32[3];
  v58 = v60;
  v59 = 0x800000000LL;
  if ( v42.m128i_i32[3] )
  {
    v15 = v60;
    v16 = v60;
    if ( v42.m128i_u32[3] > 8uLL )
    {
      sub_C8D5F0((__int64)&v58, v60, v42.m128i_u32[3], 8u, v11, v12);
      v15 = v58;
      v16 = &v58[8 * (unsigned int)v59];
    }
    for ( i = &v15[8 * v13]; i != v16; ++v16 )
    {
      if ( v16 )
        *v16 = 0;
    }
    LODWORD(v59) = v14;
    if ( v42.m128i_i32[3] )
    {
      v18 = 0;
      while ( 1 )
      {
        v25 = (unsigned __int8 *)sub_293BC00((__int64)v71, v18);
        v26 = sub_293BC00((__int64)v74, v18);
        LODWORD(v51[0]) = v18;
        v52 = 265;
        v27 = (unsigned __int8 *)v26;
        v28.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v48 = v28;
        v49.m128i_i64[0] = (__int64)".i";
        v20 = v52;
        LOWORD(v50) = 773;
        if ( (_BYTE)v52 )
        {
          if ( (_BYTE)v52 == 1 )
          {
            v34 = _mm_loadu_si128(&v49);
            v53 = _mm_loadu_si128(&v48);
            v55 = v50;
            v54 = v34;
          }
          else
          {
            if ( HIBYTE(v52) == 1 )
            {
              v19 = (_QWORD *)v51[0];
              v36 = v51[1];
            }
            else
            {
              v19 = v51;
              v20 = 2;
            }
            v54.m128i_i64[0] = (__int64)v19;
            LOBYTE(v55) = 2;
            v53.m128i_i64[0] = (__int64)&v48;
            BYTE1(v55) = v20;
            v54.m128i_i64[1] = v36;
          }
        }
        else
        {
          LOWORD(v55) = 256;
        }
        v21 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v66 + 16LL);
        v22 = **a3 - 29;
        if ( v21 == sub_9202E0 )
        {
          if ( *v25 > 0x15u || *v27 > 0x15u )
          {
LABEL_40:
            v57 = 257;
            v23 = sub_B504D0(v22, (__int64)v25, (__int64)v27, (__int64)v56, 0, 0);
            if ( (unsigned __int8)sub_920620(v23) )
            {
              v29 = v69;
              if ( v68 )
                sub_B99FD0(v23, 3u, v68);
              sub_B45150(v23, v29);
            }
            (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v67 + 16LL))(
              v67,
              v23,
              &v53,
              v64,
              v65);
            v30 = v61;
            v31 = &v61[16 * v62];
            if ( v61 != v31 )
            {
              do
              {
                v32 = *((_QWORD *)v30 + 1);
                v33 = *(_DWORD *)v30;
                v30 += 16;
                sub_B99FD0(v23, v33, v32);
              }
              while ( v31 != v30 );
            }
            goto LABEL_34;
          }
          if ( (unsigned __int8)sub_AC47B0(v22) )
            v23 = sub_AD5570(v22, (__int64)v25, v27, 0, 0);
          else
            v23 = sub_AABE40(v22, v25, v27);
        }
        else
        {
          v23 = v21(v66, v22, v25, v27);
        }
        if ( !v23 )
          goto LABEL_40;
LABEL_34:
        v24 = v18++;
        *(_QWORD *)&v58[8 * v24] = v23;
        if ( v42.m128i_i32[3] <= v18 )
        {
          v3 = a2;
          break;
        }
      }
    }
  }
  sub_293CAB0(a1, v3, (__int64)&v58, (__int64)&v42);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  if ( v75 != &v76 )
    _libc_free((unsigned __int64)v75);
  if ( v72 != &v73 )
    _libc_free((unsigned __int64)v72);
  nullsub_61();
  v70 = &unk_49DA100;
  nullsub_63();
  if ( v61 != &v63 )
    _libc_free((unsigned __int64)v61);
  return v35;
}
