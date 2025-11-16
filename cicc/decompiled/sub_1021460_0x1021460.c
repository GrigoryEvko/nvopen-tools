// Function: sub_1021460
// Address: 0x1021460
//
__int64 __fastcall sub_1021460(
        unsigned __int8 *a1,
        unsigned __int8 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int8 *v6; // rdi
  __int64 i; // r13
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // eax
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rcx
  int v14; // r14d
  unsigned int v15; // r12d
  __int64 v16; // rax
  unsigned __int8 *v17; // rdi
  unsigned __int8 *v18; // rdi
  __int64 v19; // rbx
  _QWORD *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r12
  __int64 v26; // r9
  int v27; // r11d
  __int64 v28; // r8
  _QWORD *v29; // r10
  unsigned int v30; // edi
  _QWORD *v31; // rcx
  __int64 v32; // rdx
  int v33; // eax
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int8 *v36; // r13
  int v37; // eax
  unsigned __int64 v38; // rdx
  unsigned int v40; // edx
  __int64 v41; // rsi
  _QWORD *v42; // rdi
  unsigned int v43; // r13d
  __int64 v44; // rcx
  __int64 v49; // [rsp+40h] [rbp-110h]
  __int64 v50; // [rsp+50h] [rbp-100h]
  unsigned __int8 *v51; // [rsp+58h] [rbp-F8h] BYREF
  __m128i v52; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v53; // [rsp+70h] [rbp-E0h]
  __int64 v54; // [rsp+78h] [rbp-D8h]
  __int64 v55; // [rsp+80h] [rbp-D0h]
  __int64 v56; // [rsp+88h] [rbp-C8h]
  __int64 v57; // [rsp+90h] [rbp-C0h]
  __int64 v58; // [rsp+98h] [rbp-B8h]
  __int16 v59; // [rsp+A0h] [rbp-B0h]
  __int64 v60; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-98h]
  __int64 v62; // [rsp+C0h] [rbp-90h]
  __int64 v63; // [rsp+C8h] [rbp-88h]
  _QWORD *v64; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v65; // [rsp+D8h] [rbp-78h]
  _BYTE v66[112]; // [rsp+E0h] [rbp-70h] BYREF

  v64 = v66;
  v51 = a1;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v65 = 0x800000000LL;
  v49 = sub_B43CC0((__int64)a1);
  if ( a2 )
  {
    v6 = v51;
    for ( i = *((_QWORD *)v51 + 2); i; i = *(_QWORD *)(i + 8) )
    {
      if ( v6 != *(unsigned __int8 **)(i + 24) )
      {
        v52.m128i_i64[0] = *(_QWORD *)(i + 24);
        sub_10211B0((__int64)&v60, v52.m128i_i64);
        v6 = v51;
      }
    }
    sub_BD84D0((__int64)v6, (__int64)a2);
    v10 = *v51;
    v11 = (unsigned int)(v10 - 39);
    if ( (unsigned int)v11 > 0x38 || (v12 = 0x100060000000001LL, !_bittest64(&v12, v11)) )
    {
      if ( (unsigned int)(v10 - 30) > 0xA && !(unsigned __int8)sub_B46970(v51) )
        sub_B43D60(v51);
    }
    if ( (_DWORD)v65 )
    {
LABEL_10:
      v13 = (__int64)&v64;
      v14 = 0;
      v15 = 0;
      v16 = 0;
      while ( 1 )
      {
        a2 = (unsigned __int8 **)&v52;
        v17 = (unsigned __int8 *)v64[v16];
        v53 = 0;
        v56 = 0;
        v52.m128i_i64[0] = v49;
        v51 = v17;
        v52.m128i_i64[1] = a3;
        v57 = 0;
        v54 = a4;
        v58 = 0;
        v55 = a5;
        v59 = 257;
        v50 = sub_1020E10((__int64)v17, &v52, v64, v13, v8, v9);
        if ( !v50 )
        {
          if ( a6 )
          {
            a2 = &v51;
            sub_10211B0(a6, (__int64 *)&v51);
          }
          goto LABEL_36;
        }
        v18 = v51;
        v19 = *((_QWORD *)v51 + 2);
        if ( v19 )
          break;
LABEL_31:
        a2 = (unsigned __int8 **)v50;
        sub_BD84D0((__int64)v18, v50);
        v36 = v51;
        v37 = *v51;
        v38 = (unsigned int)(v37 - 39);
        if ( (unsigned int)v38 <= 0x38 && (v13 = 0x100060000000001LL, _bittest64(&v13, v38)) )
        {
          v15 = 1;
        }
        else
        {
          v15 = 1;
          if ( (unsigned int)(v37 - 30) > 0xA )
          {
            v15 = sub_B46970(v51);
            if ( !(_BYTE)v15 )
            {
              v15 = 1;
              sub_B43D60(v36);
            }
          }
        }
LABEL_36:
        v16 = (unsigned int)(v14 + 1);
        v14 = v16;
        if ( (_DWORD)v16 == (_DWORD)v65 )
          goto LABEL_37;
      }
      while ( 1 )
      {
        v25 = *(_QWORD *)(v19 + 24);
        v52.m128i_i64[0] = v25;
        if ( !(_DWORD)v62 )
          break;
        if ( !(_DWORD)v63 )
        {
          ++v60;
          goto LABEL_45;
        }
        v26 = (unsigned int)(v63 - 1);
        v27 = 1;
        v28 = v61;
        v29 = 0;
        v30 = v26 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v31 = (_QWORD *)(v61 + 8LL * v30);
        v32 = *v31;
        if ( v25 == *v31 )
        {
LABEL_16:
          v19 = *(_QWORD *)(v19 + 8);
          if ( !v19 )
            goto LABEL_30;
        }
        else
        {
          while ( v32 != -4096 )
          {
            if ( v32 != -8192 || v29 )
              v31 = v29;
            v30 = v26 & (v27 + v30);
            v32 = *(_QWORD *)(v61 + 8LL * v30);
            if ( v25 == v32 )
              goto LABEL_16;
            ++v27;
            v29 = v31;
            v31 = (_QWORD *)(v61 + 8LL * v30);
          }
          if ( !v29 )
            v29 = v31;
          v33 = v62 + 1;
          ++v60;
          if ( 4 * ((int)v62 + 1) < (unsigned int)(3 * v63) )
          {
            if ( (int)v63 - HIDWORD(v62) - v33 <= (unsigned int)v63 >> 3 )
            {
              sub_CF4090((__int64)&v60, v63);
              if ( !(_DWORD)v63 )
              {
LABEL_76:
                LODWORD(v62) = v62 + 1;
                BUG();
              }
              v28 = 1;
              v42 = 0;
              v43 = (v63 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v29 = (_QWORD *)(v61 + 8LL * v43);
              v44 = *v29;
              v33 = v62 + 1;
              if ( v25 != *v29 )
              {
                while ( v44 != -4096 )
                {
                  if ( !v42 && v44 == -8192 )
                    v42 = v29;
                  v26 = (unsigned int)(v28 + 1);
                  v43 = (v63 - 1) & (v28 + v43);
                  v29 = (_QWORD *)(v61 + 8LL * v43);
                  v44 = *v29;
                  if ( v25 == *v29 )
                    goto LABEL_25;
                  v28 = (unsigned int)v26;
                }
                if ( v42 )
                  v29 = v42;
              }
            }
            goto LABEL_25;
          }
LABEL_45:
          sub_CF4090((__int64)&v60, 2 * v63);
          if ( !(_DWORD)v63 )
            goto LABEL_76;
          v40 = (v63 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v29 = (_QWORD *)(v61 + 8LL * v40);
          v41 = *v29;
          v33 = v62 + 1;
          if ( v25 != *v29 )
          {
            v26 = 1;
            v28 = 0;
            while ( v41 != -4096 )
            {
              if ( !v28 && v41 == -8192 )
                v28 = (__int64)v29;
              v40 = (v63 - 1) & (v26 + v40);
              v29 = (_QWORD *)(v61 + 8LL * v40);
              v41 = *v29;
              if ( v25 == *v29 )
                goto LABEL_25;
              v26 = (unsigned int)(v26 + 1);
            }
            if ( v28 )
              v29 = (_QWORD *)v28;
          }
LABEL_25:
          LODWORD(v62) = v33;
          if ( *v29 != -4096 )
            --HIDWORD(v62);
          *v29 = v25;
          v34 = (unsigned int)v65;
          v35 = (unsigned int)v65 + 1LL;
          if ( v35 > HIDWORD(v65) )
          {
            sub_C8D5F0((__int64)&v64, v66, v35, 8u, v28, v26);
            v34 = (unsigned int)v65;
          }
          v64[v34] = v25;
          LODWORD(v65) = v65 + 1;
          v19 = *(_QWORD *)(v19 + 8);
          if ( !v19 )
          {
LABEL_30:
            v18 = v51;
            goto LABEL_31;
          }
        }
      }
      v20 = &v64[(unsigned int)v65];
      if ( v20 == sub_FFE9D0(v64, (__int64)v20, v52.m128i_i64) )
        sub_1020F30((__int64)&v60, v25, v21, v22, v23, v24);
      goto LABEL_16;
    }
  }
  else
  {
    a2 = &v51;
    sub_10211B0((__int64)&v60, (__int64 *)&v51);
    if ( (_DWORD)v65 )
      goto LABEL_10;
  }
  v15 = 0;
LABEL_37:
  if ( v64 != (_QWORD *)v66 )
    _libc_free(v64, a2);
  sub_C7D6A0(v61, 8LL * (unsigned int)v63, 8);
  return v15;
}
