// Function: sub_EC7E60
// Address: 0xec7e60
//
__int64 __fastcall sub_EC7E60(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD *v12; // r14
  __int64 *v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r8
  __int64 (*v16)(void); // rax
  _BYTE *v17; // rdi
  size_t v18; // rdx
  unsigned __int8 *v19; // rsi
  _BYTE *v20; // rax
  __int64 *v21; // rax
  unsigned int v22; // eax
  __int64 v23; // r8
  _BYTE *v24; // rax
  __int64 v25; // rdi
  const char *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  _BYTE *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r11
  char v33; // al
  _QWORD *v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdi
  __m128i v37; // xmm1
  __int64 v38; // [rsp+0h] [rbp-170h]
  __int64 v39; // [rsp+0h] [rbp-170h]
  _BYTE *v40; // [rsp+8h] [rbp-168h]
  __int64 v41; // [rsp+8h] [rbp-168h]
  size_t v42; // [rsp+8h] [rbp-168h]
  __int64 v43; // [rsp+10h] [rbp-160h]
  int v44; // [rsp+10h] [rbp-160h]
  __int64 v45; // [rsp+10h] [rbp-160h]
  __int64 v46; // [rsp+18h] [rbp-158h]
  int v47; // [rsp+20h] [rbp-150h] BYREF
  __int64 v48; // [rsp+28h] [rbp-148h]
  _QWORD v49[2]; // [rsp+30h] [rbp-140h] BYREF
  __int64 v50; // [rsp+40h] [rbp-130h] BYREF
  _QWORD v51[4]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v52; // [rsp+70h] [rbp-100h]
  _QWORD v53[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v54; // [rsp+A0h] [rbp-D0h]
  __m128i v55; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i v56; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v57; // [rsp+D0h] [rbp-A0h]
  _QWORD v58[4]; // [rsp+E0h] [rbp-90h] BYREF
  char v59; // [rsp+100h] [rbp-70h]
  char v60; // [rsp+101h] [rbp-6Fh]
  __m128i v61; // [rsp+110h] [rbp-60h] BYREF
  __m128i v62; // [rsp+120h] [rbp-50h]
  __int64 v63; // [rsp+130h] [rbp-40h]

  v46 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 200LL))(*(_QWORD *)(a1 + 8));
  v5 = v4;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    v25 = *(_QWORD *)(a1 + 8);
    v61.m128i_i64[0] = (__int64)"unexpected token in '.secure_log_unique' directive";
    LOWORD(v63) = 259;
    return (unsigned int)sub_ECE0E0(v25, &v61, 0, 0);
  }
  v6 = *(unsigned __int8 *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8))
                          + 1520);
  if ( (_BYTE)v6 )
  {
    BYTE1(v63) = 1;
    v27 = ".secure_log_unique specified multiple times";
LABEL_16:
    v28 = *(_QWORD *)(a1 + 8);
    v61.m128i_i64[0] = (__int64)v27;
    LOBYTE(v63) = 3;
    return (unsigned int)sub_ECDA70(v28, a2, &v61, 0, 0);
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  if ( !*(_QWORD *)(v7 + 1488) )
  {
    BYTE1(v63) = 1;
    v27 = ".secure_log_unique used but AS_SECURE_LOG_FILE environment variable unset.";
    goto LABEL_16;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v43 = *(_QWORD *)(v7 + 1488);
  v40 = *(_BYTE **)(v7 + 1480);
  v12 = *(_QWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 48LL))(v8) + 1512);
  if ( v12 )
    goto LABEL_5;
  v47 = 0;
  v48 = sub_2241E40(v8, a2, v9, v10, v11);
  v31 = sub_22077B0(96);
  v32 = v43;
  v12 = (_QWORD *)v31;
  if ( v31 )
  {
    sub_CB7060(v31, v40, v43, (__int64)&v47, 7u);
    v32 = v43;
  }
  v45 = v32;
  if ( !v47 )
  {
    v35 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v36 = *(_QWORD *)(v35 + 1512);
    *(_QWORD *)(v35 + 1512) = v12;
    if ( v36 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
LABEL_5:
    v13 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(*(_QWORD *)(a1 + 8));
    v44 = sub_C8ED90(v13, a2);
    v14 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(*(_QWORD *)(a1 + 8));
    v15 = (__int64)v12;
    v16 = *(__int64 (**)(void))(**(_QWORD **)(*v14 + 24LL * (unsigned int)(v44 - 1)) + 16LL);
    if ( (char *)v16 == (char *)sub_C1E8B0 )
    {
      v17 = (_BYTE *)v12[4];
      v18 = 14;
      v19 = (unsigned __int8 *)"Unknown buffer";
      if ( v12[3] - (_QWORD)v17 <= 0xDu )
      {
LABEL_7:
        v15 = sub_CB6200((__int64)v12, v19, v18);
        v20 = *(_BYTE **)(v15 + 24);
        v17 = *(_BYTE **)(v15 + 32);
LABEL_8:
        if ( v17 == v20 )
        {
          v15 = sub_CB6200(v15, (unsigned __int8 *)":", 1u);
        }
        else
        {
          *v17 = 58;
          ++*(_QWORD *)(v15 + 32);
        }
        v41 = v15;
        v21 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(*(_QWORD *)(a1 + 8));
        v22 = sub_C90410(v21, a2, v44);
        v23 = sub_CB59D0(v41, v22);
        v24 = *(_BYTE **)(v23 + 32);
        if ( *(_BYTE **)(v23 + 24) == v24 )
        {
          v23 = sub_CB6200(v23, (unsigned __int8 *)":", 1u);
        }
        else
        {
          *v24 = 58;
          ++*(_QWORD *)(v23 + 32);
        }
        v61.m128i_i64[1] = v5;
        LOWORD(v63) = 773;
        v61.m128i_i64[0] = v46;
        v62.m128i_i64[0] = (__int64)"\n";
        sub_CA0E80((__int64)&v61, v23);
        *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 1520) = 1;
        return v6;
      }
    }
    else
    {
      v29 = v16();
      v17 = (_BYTE *)v12[4];
      v15 = (__int64)v12;
      v19 = (unsigned __int8 *)v29;
      v20 = (_BYTE *)v12[3];
      if ( v18 > v20 - v17 )
        goto LABEL_7;
      if ( !v18 )
        goto LABEL_8;
    }
    v39 = v15;
    v42 = v18;
    memcpy(v17, v19, v18);
    v30 = (_BYTE *)(v12[4] + v42);
    v12[4] = v30;
    v20 = (_BYTE *)v12[3];
    v17 = v30;
    v15 = v39;
    goto LABEL_8;
  }
  v60 = 1;
  v58[0] = ")";
  v59 = 3;
  (*(void (__fastcall **)(_QWORD *))(*(_QWORD *)v48 + 32LL))(v49);
  v51[0] = "can't open secure log file: ";
  v51[3] = v45;
  v51[2] = v40;
  v53[0] = v51;
  v53[2] = " (";
  v55.m128i_i64[0] = (__int64)v53;
  v33 = v59;
  v52 = 1283;
  v54 = 770;
  v56.m128i_i64[0] = (__int64)v49;
  LOWORD(v57) = 1026;
  if ( v59 )
  {
    if ( v59 == 1 )
    {
      v37 = _mm_loadu_si128(&v56);
      v61 = _mm_loadu_si128(&v55);
      v63 = v57;
      v62 = v37;
    }
    else
    {
      if ( v60 == 1 )
      {
        v34 = (_QWORD *)v58[0];
        v38 = v58[1];
      }
      else
      {
        v34 = v58;
        v33 = 2;
      }
      v62.m128i_i64[0] = (__int64)v34;
      v61.m128i_i64[0] = (__int64)&v55;
      LOBYTE(v63) = 2;
      v62.m128i_i64[1] = v38;
      BYTE1(v63) = v33;
    }
  }
  else
  {
    LOWORD(v63) = 256;
  }
  v6 = sub_ECDA70(*(_QWORD *)(a1 + 8), a2, &v61, 0, 0);
  if ( (__int64 *)v49[0] != &v50 )
    j_j___libc_free_0(v49[0], v50 + 1);
  if ( v12 )
    (*(void (__fastcall **)(_QWORD *))(*v12 + 8LL))(v12);
  return v6;
}
