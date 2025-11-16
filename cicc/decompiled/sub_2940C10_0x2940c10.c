// Function: sub_2940C10
// Address: 0x2940c10
//
__int64 __fastcall sub_2940C10(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned int v4; // r13d
  __int64 v6; // r15
  _QWORD *v7; // r14
  _BYTE *v8; // r15
  __int64 v9; // rsi
  __int64 v10; // rdx
  int v11; // eax
  _QWORD *v12; // rdi
  __int64 v13; // rax
  unsigned __int8 *v14; // r14
  __int64 (__fastcall *v15)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned int v19; // r14d
  __int64 v20; // r12
  _QWORD *v21; // rdx
  char v22; // al
  __int64 v23; // rax
  __m128i v24; // rax
  char v25; // al
  _QWORD *v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // rax
  __m128i v29; // rax
  _QWORD *v30; // rax
  unsigned int *v31; // r15
  unsigned int *v32; // rbx
  __int64 v33; // r14
  __int64 v34; // rdx
  unsigned int v35; // esi
  __m128i v36; // xmm1
  __m128i v37; // xmm3
  __int64 v38; // rax
  __int64 v39; // r9
  unsigned __int64 *v40; // r14
  _QWORD *v41; // rdi
  unsigned __int64 v42; // rdi
  __int32 v43; // ebx
  __int64 v44; // [rsp+20h] [rbp-250h]
  __int64 v45; // [rsp+28h] [rbp-248h]
  __int64 v46; // [rsp+38h] [rbp-238h]
  unsigned __int32 v47; // [rsp+40h] [rbp-230h]
  __int64 v48; // [rsp+40h] [rbp-230h]
  unsigned __int32 v49; // [rsp+48h] [rbp-228h]
  _QWORD *v50; // [rsp+48h] [rbp-228h]
  _QWORD *v51; // [rsp+48h] [rbp-228h]
  __m128i v53; // [rsp+50h] [rbp-220h] BYREF
  __int64 v54; // [rsp+68h] [rbp-208h]
  unsigned __int8 v55; // [rsp+70h] [rbp-200h]
  __m128i v56; // [rsp+80h] [rbp-1F0h] BYREF
  __m128i v57; // [rsp+90h] [rbp-1E0h] BYREF
  __int64 v58; // [rsp+A0h] [rbp-1D0h]
  _QWORD *v59; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v60; // [rsp+B8h] [rbp-1B8h]
  __int16 v61; // [rsp+D0h] [rbp-1A0h]
  __m128i v62; // [rsp+E0h] [rbp-190h] BYREF
  __m128i v63; // [rsp+F0h] [rbp-180h]
  __int64 v64; // [rsp+100h] [rbp-170h]
  unsigned int *v65; // [rsp+110h] [rbp-160h] BYREF
  int v66; // [rsp+118h] [rbp-158h]
  char v67; // [rsp+120h] [rbp-150h] BYREF
  __int64 v68; // [rsp+148h] [rbp-128h]
  __int64 v69; // [rsp+150h] [rbp-120h]
  _QWORD *v70; // [rsp+158h] [rbp-118h]
  __int64 v71; // [rsp+160h] [rbp-110h]
  __int64 v72; // [rsp+168h] [rbp-108h]
  void *v73; // [rsp+190h] [rbp-E0h]
  __m128i v74[5]; // [rsp+1A0h] [rbp-D0h] BYREF
  char *v75; // [rsp+1F0h] [rbp-80h]
  char v76; // [rsp+200h] [rbp-70h] BYREF

  v2 = a1;
  v3 = (__int64)a2;
  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, a2) )
    return 0;
  sub_2939E80((__int64)&v53, a1, *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL));
  v4 = v55;
  if ( !v55 )
    return 0;
  sub_23D0AB0((__int64)&v65, (__int64)a2, 0, 0, 0);
  sub_293CE40(v74, (_QWORD *)a1, (__int64)a2, *((_QWORD *)a2 - 8), &v53);
  v6 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_QWORD **)(v6 + 24);
    if ( *(_DWORD *)(v6 + 32) > 0x40u )
      v7 = (_QWORD *)*v7;
    v47 = (unsigned int)v7 / v53.m128i_i32[2];
    v8 = (_BYTE *)sub_293BC00((__int64)v74, (unsigned int)v7 / v53.m128i_i32[2]);
    if ( v53.m128i_i32[3] - 1 == v47 && v54 && (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 > 1
      || v53.m128i_i32[2] <= 1u )
    {
LABEL_10:
      v9 = (__int64)v8;
      if ( (_BYTE *)v3 != v8 )
      {
LABEL_11:
        sub_BD84D0(v3, v9);
        v10 = *(unsigned int *)(v2 + 336);
        v11 = v10;
        if ( *(_DWORD *)(v2 + 340) <= (unsigned int)v10 )
        {
          v38 = sub_C8D7D0(v2 + 328, v2 + 344, 0, 0x18u, (unsigned __int64 *)&v62, v2 + 328);
          v39 = v2 + 328;
          v40 = (unsigned __int64 *)v38;
          v41 = (_QWORD *)(v38 + 24LL * *(unsigned int *)(v2 + 336));
          if ( v41 )
          {
            *v41 = 6;
            v41[1] = 0;
            v41[2] = v3;
            if ( v3 != -4096 && v3 != -8192 )
            {
              sub_BD73F0((__int64)v41);
              v39 = v2 + 328;
            }
          }
          sub_F17F80(v39, v40);
          v42 = *(_QWORD *)(v2 + 328);
          v43 = v62.m128i_i32[0];
          if ( v2 + 344 != v42 )
            _libc_free(v42);
          ++*(_DWORD *)(v2 + 336);
          *(_QWORD *)(v2 + 328) = v40;
          *(_DWORD *)(v2 + 340) = v43;
        }
        else
        {
          v12 = (_QWORD *)(*(_QWORD *)(v2 + 328) + 24 * v10);
          if ( v12 )
          {
            *v12 = 6;
            v12[1] = 0;
            v12[2] = v3;
            if ( v3 != -8192 && v3 != -4096 )
              sub_BD73F0((__int64)v12);
            v11 = *(_DWORD *)(v2 + 336);
          }
          *(_DWORD *)(v2 + 336) = v11 + 1;
        }
        *(_BYTE *)(v2 + 320) = 1;
        goto LABEL_19;
      }
      goto LABEL_19;
    }
    v49 = v53.m128i_u32[2];
    v61 = 257;
    v13 = sub_BCB2E0(v70);
    v14 = (unsigned __int8 *)sub_ACD640(v13, (unsigned int)v7 % v49, 0);
    v15 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v71 + 96LL);
    if ( v15 == sub_948070 )
    {
      if ( *v8 > 0x15u || *v14 > 0x15u )
        goto LABEL_54;
      v16 = sub_AD5840((__int64)v8, v14, 0);
    }
    else
    {
      v16 = v15(v71, v8, v14);
    }
    if ( v16 )
    {
LABEL_33:
      v8 = (_BYTE *)v16;
      goto LABEL_10;
    }
LABEL_54:
    LOWORD(v64) = 257;
    v30 = sub_BD2C40(72, 2u);
    if ( v30 )
    {
      v50 = v30;
      sub_B4DE80((__int64)v30, (__int64)v8, (__int64)v14, (__int64)&v62, 0, 0);
      v30 = v50;
    }
    v51 = v30;
    (*(void (__fastcall **)(__int64, _QWORD *, _QWORD **, __int64, __int64))(*(_QWORD *)v72 + 16LL))(
      v72,
      v30,
      &v59,
      v68,
      v69);
    v16 = (__int64)v51;
    v31 = &v65[4 * v66];
    if ( v65 != v31 )
    {
      v32 = v65;
      v33 = v16;
      do
      {
        v34 = *((_QWORD *)v32 + 1);
        v35 = *v32;
        v32 += 4;
        sub_B99FD0(v33, v35, v34);
      }
      while ( v31 != v32 );
      v3 = (__int64)a2;
      v16 = v33;
    }
    goto LABEL_33;
  }
  if ( *(_BYTE *)(a1 + 1128) && v53.m128i_i32[2] <= 1u )
  {
    v17 = sub_ACADE0(*(__int64 ***)(v53.m128i_i64[0] + 24));
    v18 = v17;
    if ( v53.m128i_i32[3] )
    {
      v19 = 0;
      v20 = v17;
      do
      {
        LODWORD(v59) = v19;
        v61 = 265;
        v24.m128i_i64[0] = (__int64)sub_BD5D20(v6);
        v56 = v24;
        v57.m128i_i64[0] = (__int64)".is.";
        v25 = v61;
        LOWORD(v58) = 773;
        if ( (_BYTE)v61 )
        {
          if ( (_BYTE)v61 == 1 )
          {
            v36 = _mm_loadu_si128(&v57);
            v62 = _mm_loadu_si128(&v56);
            v64 = v58;
            v63 = v36;
          }
          else
          {
            if ( HIBYTE(v61) == 1 )
            {
              v26 = v59;
              v45 = v60;
            }
            else
            {
              v26 = &v59;
              v25 = 2;
            }
            v63.m128i_i64[0] = (__int64)v26;
            LOBYTE(v64) = 2;
            v62.m128i_i64[0] = (__int64)&v56;
            BYTE1(v64) = v25;
            v63.m128i_i64[1] = v45;
          }
        }
        else
        {
          LOWORD(v64) = 256;
        }
        v27 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v6 + 8), v19, 0);
        v46 = sub_92B530(&v65, 0x20u, v6, v27, (__int64)&v62);
        v28 = sub_293BC00((__int64)v74, v19);
        LODWORD(v59) = v19;
        v48 = v28;
        v61 = 265;
        v29.m128i_i64[0] = (__int64)sub_BD5D20((__int64)a2);
        v56 = v29;
        v57.m128i_i64[0] = (__int64)".upto";
        v22 = v61;
        LOWORD(v58) = 773;
        if ( (_BYTE)v61 )
        {
          if ( (_BYTE)v61 == 1 )
          {
            v37 = _mm_loadu_si128(&v57);
            v62 = _mm_loadu_si128(&v56);
            v64 = v58;
            v63 = v37;
          }
          else
          {
            if ( HIBYTE(v61) == 1 )
            {
              v21 = v59;
              v44 = v60;
            }
            else
            {
              v21 = &v59;
              v22 = 2;
            }
            v63.m128i_i64[0] = (__int64)v21;
            LOBYTE(v64) = 2;
            v62.m128i_i64[0] = (__int64)&v56;
            BYTE1(v64) = v22;
            v63.m128i_i64[1] = v44;
          }
        }
        else
        {
          LOWORD(v64) = 256;
        }
        ++v19;
        v23 = sub_B36550(&v65, v46, v48, v20, (__int64)&v62, 0);
        v20 = v23;
      }
      while ( v53.m128i_i32[3] > v19 );
      v4 = (unsigned __int8)v4;
      v2 = a1;
      v18 = v23;
    }
    v9 = v18;
    if ( v3 == v18 )
      goto LABEL_19;
    goto LABEL_11;
  }
  v4 = 0;
LABEL_19:
  if ( v75 != &v76 )
    _libc_free((unsigned __int64)v75);
  nullsub_61();
  v73 = &unk_49DA100;
  nullsub_63();
  if ( v65 != (unsigned int *)&v67 )
    _libc_free((unsigned __int64)v65);
  return v4;
}
