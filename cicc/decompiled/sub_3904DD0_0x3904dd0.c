// Function: sub_3904DD0
// Address: 0x3904dd0
//
__int64 __fastcall sub_3904DD0(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r12d
  char *v10; // r15
  _QWORD *v11; // r14
  __int64 v12; // r15
  __int64 *v13; // rax
  __int64 (*v14)(void); // rax
  _BYTE *v15; // rdi
  size_t v16; // rdx
  char *v17; // rsi
  _BYTE *v18; // rax
  __int64 *v19; // rax
  unsigned int v20; // eax
  __int64 v21; // r8
  _BYTE *v22; // rax
  __int64 v23; // rdi
  const char *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  _BYTE *v28; // rdx
  __int64 v29; // rax
  const char *v30; // rax
  bool v31; // zf
  char v32; // dl
  char v33; // al
  char **v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdi
  size_t v37; // [rsp+0h] [rbp-120h]
  int v38; // [rsp+8h] [rbp-118h]
  size_t v39; // [rsp+8h] [rbp-118h]
  _QWORD v40[2]; // [rsp+10h] [rbp-110h] BYREF
  int v41; // [rsp+20h] [rbp-100h] BYREF
  __int64 (__fastcall **v42)(); // [rsp+28h] [rbp-F8h]
  _QWORD v43[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v44; // [rsp+40h] [rbp-E0h]
  _QWORD v45[2]; // [rsp+50h] [rbp-D0h] BYREF
  char v46; // [rsp+60h] [rbp-C0h]
  char v47; // [rsp+61h] [rbp-BFh]
  __m128i v48; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+80h] [rbp-A0h]
  char *v50; // [rsp+90h] [rbp-90h] BYREF
  char v51; // [rsp+A0h] [rbp-80h]
  char v52; // [rsp+A1h] [rbp-7Fh]
  __m128i v53; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v54; // [rsp+C0h] [rbp-60h]
  unsigned __int64 v55[2]; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v56; // [rsp+E0h] [rbp-40h] BYREF

  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 152LL))(*(_QWORD *)(a1 + 8));
  v5 = *(_QWORD *)(a1 + 8);
  v40[0] = v4;
  v40[1] = v6;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 40LL))(v5) + 8) != 9 )
  {
    v23 = *(_QWORD *)(a1 + 8);
    v55[0] = (unsigned __int64)"unexpected token in '.secure_log_unique' directive";
    LOWORD(v56) = 259;
    return (unsigned int)sub_3909CF0(v23, v55, 0, 0, v7, v8);
  }
  v9 = *(unsigned __int8 *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 744);
  if ( (_BYTE)v9 )
  {
    BYTE1(v56) = 1;
    v25 = ".secure_log_unique specified multiple times";
LABEL_16:
    v26 = *(_QWORD *)(a1 + 8);
    v55[0] = (unsigned __int64)v25;
    LOBYTE(v56) = 3;
    return (unsigned int)sub_3909790(v26, a2, v55, 0, 0);
  }
  v10 = *(char **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 728);
  if ( !v10 )
  {
    BYTE1(v56) = 1;
    v25 = ".secure_log_unique used but AS_SECURE_LOG_FILE environment variable unset.";
    goto LABEL_16;
  }
  v11 = *(_QWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 736);
  if ( v11 )
    goto LABEL_5;
  v41 = 0;
  v42 = sub_2241E40();
  v39 = strlen(v10);
  v29 = sub_22077B0(0x50u);
  v11 = (_QWORD *)v29;
  if ( v29 )
    sub_16E8AF0(v29, v10, v39, (__int64)&v41, 3u);
  if ( !v41 )
  {
    v35 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v36 = *(_QWORD *)(v35 + 736);
    *(_QWORD *)(v35 + 736) = v11;
    if ( v36 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
LABEL_5:
    v12 = (__int64)v11;
    v13 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(*(_QWORD *)(a1 + 8));
    v38 = sub_16CE270(v13, a2);
    v14 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(*(_QWORD *)(a1 + 8))
                                            + 24LL * (unsigned int)(v38 - 1))
                              + 16LL);
    if ( (char *)v14 == (char *)sub_12BCB10 )
    {
      v15 = (_BYTE *)v11[3];
      v16 = 14;
      v17 = "Unknown buffer";
      if ( v11[2] - (_QWORD)v15 <= 0xDu )
      {
LABEL_7:
        v12 = sub_16E7EE0((__int64)v11, v17, v16);
        v18 = *(_BYTE **)(v12 + 16);
        v15 = *(_BYTE **)(v12 + 24);
LABEL_8:
        if ( v15 == v18 )
        {
          v12 = sub_16E7EE0(v12, ":", 1u);
        }
        else
        {
          *v15 = 58;
          ++*(_QWORD *)(v12 + 24);
        }
        v19 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(*(_QWORD *)(a1 + 8));
        v20 = sub_16CFA40(v19, a2, v38);
        v21 = sub_16E7A90(v12, v20);
        v22 = *(_BYTE **)(v21 + 24);
        if ( *(_BYTE **)(v21 + 16) == v22 )
        {
          v21 = sub_16E7EE0(v21, ":", 1u);
        }
        else
        {
          *v22 = 58;
          ++*(_QWORD *)(v21 + 24);
        }
        LOWORD(v56) = 773;
        v55[0] = (unsigned __int64)v40;
        v55[1] = (unsigned __int64)"\n";
        sub_16E2CE0((__int64)v55, v21);
        *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 744) = 1;
        return v9;
      }
    }
    else
    {
      v27 = v14();
      v15 = (_BYTE *)v11[3];
      v17 = (char *)v27;
      v18 = (_BYTE *)v11[2];
      if ( v18 - v15 < v16 )
        goto LABEL_7;
      if ( !v16 )
        goto LABEL_8;
    }
    v37 = v16;
    memcpy(v15, v17, v16);
    v28 = (_BYTE *)(v11[3] + v37);
    v11[3] = v28;
    v18 = (_BYTE *)v11[2];
    v15 = v28;
    goto LABEL_8;
  }
  v52 = 1;
  v50 = ")";
  v51 = 3;
  (*((void (__fastcall **)(unsigned __int64 *))*v42 + 4))(v55);
  v30 = "can't open secure log file: ";
  v31 = *v10 == 0;
  v43[0] = "can't open secure log file: ";
  if ( v31 )
  {
    v32 = 3;
    v44 = 259;
  }
  else
  {
    v43[1] = v10;
    v32 = 2;
    v30 = (const char *)v43;
    v44 = 771;
  }
  v45[0] = v30;
  v45[1] = " (";
  v48.m128i_i64[0] = (__int64)v45;
  v33 = v51;
  v46 = v32;
  v47 = 3;
  v48.m128i_i64[1] = (__int64)v55;
  LOWORD(v49) = 1026;
  if ( v51 )
  {
    if ( v51 == 1 )
    {
      v53 = _mm_loadu_si128(&v48);
      v54 = v49;
    }
    else
    {
      v34 = (char **)v50;
      if ( v52 != 1 )
      {
        v34 = &v50;
        v33 = 2;
      }
      v53.m128i_i64[1] = (__int64)v34;
      v53.m128i_i64[0] = (__int64)&v48;
      LOBYTE(v54) = 2;
      BYTE1(v54) = v33;
    }
  }
  else
  {
    LOWORD(v54) = 256;
  }
  v9 = sub_3909790(*(_QWORD *)(a1 + 8), a2, &v53, 0, 0);
  if ( (__int64 *)v55[0] != &v56 )
    j_j___libc_free_0(v55[0]);
  if ( v11 )
    (*(void (__fastcall **)(_QWORD *))(*v11 + 8LL))(v11);
  return v9;
}
