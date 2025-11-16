// Function: sub_293E340
// Address: 0x293e340
//
__int64 __fastcall sub_293E340(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rbx
  unsigned int v8; // r14d
  _BYTE *v9; // rdx
  _QWORD *v10; // rax
  _QWORD *i; // rdx
  unsigned int v12; // ebx
  _QWORD *v13; // rdx
  char v14; // al
  __int64 **v15; // r12
  __int64 v16; // r14
  unsigned int v17; // r15d
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v19; // r13
  __int64 v20; // rdx
  __m128i v21; // rax
  int v22; // r14d
  char *v23; // r14
  char *v24; // r15
  __int64 v25; // rdx
  unsigned int v26; // esi
  __m128i v27; // xmm1
  __int64 v28; // [rsp+38h] [rbp-2E8h]
  unsigned __int8 v30; // [rsp+4Fh] [rbp-2D1h]
  char v31[8]; // [rsp+50h] [rbp-2D0h] BYREF
  int v32; // [rsp+58h] [rbp-2C8h]
  unsigned int v33; // [rsp+5Ch] [rbp-2C4h]
  __int64 **v34; // [rsp+60h] [rbp-2C0h]
  __int64 **v35; // [rsp+68h] [rbp-2B8h]
  char v36; // [rsp+70h] [rbp-2B0h]
  __m128i v37[2]; // [rsp+80h] [rbp-2A0h] BYREF
  unsigned __int8 v38; // [rsp+A0h] [rbp-280h]
  __m128i v39; // [rsp+B0h] [rbp-270h] BYREF
  __m128i v40; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v41; // [rsp+D0h] [rbp-250h]
  _QWORD v42[4]; // [rsp+E0h] [rbp-240h] BYREF
  __int16 v43; // [rsp+100h] [rbp-220h]
  __m128i v44; // [rsp+110h] [rbp-210h] BYREF
  __m128i v45; // [rsp+120h] [rbp-200h]
  __int64 v46; // [rsp+130h] [rbp-1F0h]
  char v47[32]; // [rsp+140h] [rbp-1E0h] BYREF
  __int16 v48; // [rsp+160h] [rbp-1C0h]
  _BYTE *v49; // [rsp+170h] [rbp-1B0h] BYREF
  __int64 v50; // [rsp+178h] [rbp-1A8h]
  _BYTE v51[64]; // [rsp+180h] [rbp-1A0h] BYREF
  char *v52; // [rsp+1C0h] [rbp-160h] BYREF
  int v53; // [rsp+1C8h] [rbp-158h]
  char v54; // [rsp+1D0h] [rbp-150h] BYREF
  __int64 v55; // [rsp+1F8h] [rbp-128h]
  __int64 v56; // [rsp+200h] [rbp-120h]
  __int64 v57; // [rsp+210h] [rbp-110h]
  __int64 v58; // [rsp+218h] [rbp-108h]
  __int64 v59; // [rsp+220h] [rbp-100h]
  int v60; // [rsp+228h] [rbp-F8h]
  void *v61; // [rsp+240h] [rbp-E0h]
  __m128i v62[5]; // [rsp+250h] [rbp-D0h] BYREF
  char *v63; // [rsp+2A0h] [rbp-80h]
  char v64; // [rsp+2B0h] [rbp-70h] BYREF

  v2 = a1;
  v3 = (unsigned __int64)a2;
  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, a2) )
    return 0;
  sub_2939E80((__int64)v31, a1, *((_QWORD *)a2 + 1));
  if ( !v36 )
    return 0;
  sub_2939E80((__int64)v37, a1, *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL));
  v30 = v38;
  if ( !v38 || v37[0].m128i_i32[2] != v32 )
    return 0;
  sub_23D0AB0((__int64)&v52, (__int64)a2, 0, 0, 0);
  sub_293CE40(v62, (_QWORD *)a1, (__int64)a2, *((_QWORD *)a2 - 4), v37);
  v7 = v33;
  v8 = v33;
  v49 = v51;
  v50 = 0x800000000LL;
  if ( v33 )
  {
    v9 = v51;
    v10 = v51;
    if ( v33 > 8uLL )
    {
      sub_C8D5F0((__int64)&v49, v51, v33, 8u, v5, v6);
      v9 = v49;
      v10 = &v49[8 * (unsigned int)v50];
    }
    for ( i = &v9[8 * v7]; i != v10; ++v10 )
    {
      if ( v10 )
        *v10 = 0;
    }
    LODWORD(v50) = v8;
    if ( v33 )
    {
      v12 = 0;
      while ( 1 )
      {
        LODWORD(v42[0]) = v12;
        v43 = 265;
        v21.m128i_i64[0] = (__int64)sub_BD5D20((__int64)a2);
        v39 = v21;
        v40.m128i_i64[0] = (__int64)".i";
        v14 = v43;
        LOWORD(v41) = 773;
        if ( (_BYTE)v43 )
        {
          if ( (_BYTE)v43 == 1 )
          {
            v27 = _mm_loadu_si128(&v40);
            v44 = _mm_loadu_si128(&v39);
            v46 = v41;
            v45 = v27;
          }
          else
          {
            if ( HIBYTE(v43) == 1 )
            {
              v13 = (_QWORD *)v42[0];
              v28 = v42[1];
            }
            else
            {
              v13 = v42;
              v14 = 2;
            }
            v45.m128i_i64[0] = (__int64)v13;
            LOBYTE(v46) = 2;
            v44.m128i_i64[0] = (__int64)&v39;
            BYTE1(v46) = v14;
            v45.m128i_i64[1] = v28;
          }
        }
        else
        {
          LOWORD(v46) = 256;
        }
        v15 = v35;
        if ( !v35 || v33 - 1 != v12 )
          v15 = v34;
        v16 = sub_293BC00((__int64)v62, v12);
        v17 = *a2 - 29;
        if ( v15 == *(__int64 ***)(v16 + 8) )
        {
          v19 = v16;
        }
        else
        {
          v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v57 + 120LL);
          if ( v18 == sub_920130 )
          {
            if ( *(_BYTE *)v16 > 0x15u )
              goto LABEL_34;
            if ( (unsigned __int8)sub_AC4810(v17) )
              v19 = sub_ADAB70(v17, v16, v15, 0);
            else
              v19 = sub_AA93C0(v17, v16, (__int64)v15);
          }
          else
          {
            v19 = v18(v57, v17, (_BYTE *)v16, (__int64)v15);
          }
          if ( !v19 )
          {
LABEL_34:
            v48 = 257;
            v19 = sub_B51D30(v17, v16, (__int64)v15, (__int64)v47, 0, 0);
            if ( (unsigned __int8)sub_920620(v19) )
            {
              v22 = v60;
              if ( v59 )
                sub_B99FD0(v19, 3u, v59);
              sub_B45150(v19, v22);
            }
            (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
              v58,
              v19,
              &v44,
              v55,
              v56);
            v23 = &v52[16 * v53];
            if ( v52 != v23 )
            {
              v24 = v52;
              do
              {
                v25 = *((_QWORD *)v24 + 1);
                v26 = *(_DWORD *)v24;
                v24 += 16;
                sub_B99FD0(v19, v26, v25);
              }
              while ( v23 != v24 );
            }
          }
        }
        v20 = v12++;
        *(_QWORD *)&v49[8 * v20] = v19;
        if ( v33 <= v12 )
        {
          v2 = a1;
          v3 = (unsigned __int64)a2;
          break;
        }
      }
    }
  }
  sub_293CAB0(v2, v3, (__int64)&v49, (__int64)v31);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( v63 != &v64 )
    _libc_free((unsigned __int64)v63);
  nullsub_61();
  v61 = &unk_49DA100;
  nullsub_63();
  if ( v52 != &v54 )
    _libc_free((unsigned __int64)v52);
  return v30;
}
