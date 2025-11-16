// Function: sub_26A8EB0
// Address: 0x26a8eb0
//
__int64 __fastcall sub_26A8EB0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  char v5; // al
  __int64 (__fastcall *v6)(__int64, __int64); // rsi
  char v7; // al
  unsigned int v8; // r13d
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // r14
  unsigned __int8 *v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  _BYTE *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  _BYTE *v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  bool v28; // zf
  __int64 v29; // rax
  unsigned __int64 *v30; // rax
  unsigned __int64 *v31; // r13
  char v32; // r12
  __int64 v33; // rax
  char v34; // dl
  int v35; // [rsp+24h] [rbp-20Ch]
  unsigned __int64 *v36; // [rsp+28h] [rbp-208h]
  int v37; // [rsp+30h] [rbp-200h]
  char v38; // [rsp+37h] [rbp-1F9h]
  char v39; // [rsp+38h] [rbp-1F8h]
  char v40; // [rsp+40h] [rbp-1F0h] BYREF
  char v41; // [rsp+41h] [rbp-1EFh] BYREF
  char v42; // [rsp+42h] [rbp-1EEh] BYREF
  char v43; // [rsp+43h] [rbp-1EDh] BYREF
  __int64 v44; // [rsp+44h] [rbp-1ECh] BYREF
  int v45; // [rsp+4Ch] [rbp-1E4h]
  _QWORD v46[2]; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v47; // [rsp+60h] [rbp-1D0h] BYREF
  __int64 v48; // [rsp+68h] [rbp-1C8h]
  char *v49; // [rsp+70h] [rbp-1C0h]
  char *v50; // [rsp+78h] [rbp-1B8h]
  _QWORD v51[7]; // [rsp+80h] [rbp-1B0h] BYREF
  unsigned int v52; // [rsp+B8h] [rbp-178h]
  _QWORD *v53; // [rsp+C0h] [rbp-170h]
  _QWORD v54[5]; // [rsp+D0h] [rbp-160h] BYREF
  unsigned int v55; // [rsp+F8h] [rbp-138h]
  _QWORD *v56; // [rsp+100h] [rbp-130h]
  _QWORD v57[5]; // [rsp+110h] [rbp-120h] BYREF
  unsigned int v58; // [rsp+138h] [rbp-F8h]
  char *v59; // [rsp+140h] [rbp-F0h]
  char v60; // [rsp+150h] [rbp-E0h] BYREF
  __int64 (__fastcall **v61)(); // [rsp+170h] [rbp-C0h]
  __int64 v62; // [rsp+188h] [rbp-A8h]
  unsigned int v63; // [rsp+198h] [rbp-98h]
  _QWORD *v64; // [rsp+1A0h] [rbp-90h]
  _QWORD v65[5]; // [rsp+1B0h] [rbp-80h] BYREF
  unsigned int v66; // [rsp+1D8h] [rbp-58h]
  char *v67; // [rsp+1E0h] [rbp-50h]
  char v68; // [rsp+1F8h] [rbp-38h] BYREF

  v3 = a1 + 88;
  sub_266FF60((__int64)v51, a1 + 88);
  v46[0] = a2;
  v5 = *(_BYTE *)(a1 + 241);
  v46[1] = a1;
  v40 = 0;
  if ( *(_BYTE *)(a1 + 240) != v5
    && !(unsigned __int8)sub_25264B0(
                           a2,
                           (unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64))sub_269E480,
                           (__int64)v46,
                           a1,
                           &v40) )
  {
    v28 = *(_BYTE *)(a1 + 320) == 0;
    *(_BYTE *)(a1 + 241) = *(_BYTE *)(a1 + 240);
    if ( v28 )
      goto LABEL_35;
LABEL_3:
    v39 = 0;
    goto LABEL_4;
  }
  if ( *(_BYTE *)(a1 + 320) )
    goto LABEL_3;
LABEL_35:
  v29 = *(_QWORD *)(a2 + 208);
  v47 = a2;
  v48 = v29 + 28792;
  v49 = (char *)a1;
  LOBYTE(v44) = 1;
  if ( !(unsigned __int8)sub_2523890(
                           a2,
                           (__int64 (__fastcall *)(__int64, __int64 *))sub_26AAF60,
                           (__int64)&v47,
                           a1,
                           1u,
                           &v44) )
    *(_BYTE *)(a1 + 401) = *(_BYTE *)(a1 + 400);
  LOBYTE(v44) = 1;
  v47 = a2;
  v48 = a1;
  if ( !(unsigned __int8)sub_2523890(
                           a2,
                           (__int64 (__fastcall *)(__int64, __int64 *))sub_26AAC90,
                           (__int64)&v47,
                           a1,
                           1u,
                           &v44) )
    *(_BYTE *)(a1 + 337) = *(_BYTE *)(a1 + 336);
  v39 = v44 ^ 1;
  if ( *(_DWORD *)(a1 + 288) )
  {
    if ( !*(_BYTE *)(a1 + 401) )
      goto LABEL_53;
    v38 = *(_BYTE *)(a1 + 337);
    if ( !v38 )
      goto LABEL_53;
    v30 = *(unsigned __int64 **)(a1 + 376);
    v36 = &v30[*(unsigned int *)(a1 + 384)];
    if ( v36 != v30 )
    {
      v31 = *(unsigned __int64 **)(a1 + 376);
      v32 = v44 ^ 1;
      v37 = 0;
      v35 = 0;
      do
      {
        sub_250D230((unsigned __int64 *)&v47, *v31, 4, 0);
        v33 = sub_26A73D0(a2, v47, v48, a1, 1, 1);
        if ( v33 )
        {
          v34 = *(_BYTE *)(v33 + 241);
          if ( v34 )
            ++v35;
          else
            ++v37;
          if ( v34 != *(_BYTE *)(v33 + 240) )
            v32 = v38;
        }
        else
        {
          ++v37;
          v32 = v38;
        }
        ++v31;
      }
      while ( v36 != v31 );
      v39 = v32;
      v3 = a1 + 88;
      if ( v35 )
      {
        if ( v37 )
LABEL_53:
          *(_BYTE *)(a1 + 241) = *(_BYTE *)(a1 + 240);
      }
    }
  }
LABEL_4:
  v49 = &v42;
  v6 = sub_26A8EA0;
  v50 = &v41;
  v44 = 0xB00000005LL;
  v41 = 1;
  v42 = 1;
  v47 = a2;
  v48 = a1;
  v43 = 0;
  v45 = 56;
  if ( (unsigned __int8)sub_2526370(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_26A8EA0,
                          (__int64)&v47,
                          a1,
                          (int *)&v44,
                          3,
                          &v43,
                          0,
                          0) )
  {
    if ( !v43 )
    {
      if ( v41 )
      {
        *(_BYTE *)(a1 + 112) = *(_BYTE *)(a1 + 113);
        *(_BYTE *)(a1 + 176) = *(_BYTE *)(a1 + 177);
      }
      if ( !v40 && !v39 && v42 )
        *(_BYTE *)(a1 + 240) = *(_BYTE *)(a1 + 241);
    }
    v6 = (__int64 (__fastcall *)(__int64, __int64))v3;
    v8 = (unsigned __int8)sub_266F260((__int64)v51, v3);
  }
  else
  {
    v7 = *(_BYTE *)(a1 + 400);
    *(_BYTE *)(a1 + 96) = 1;
    v8 = 0;
    *(_BYTE *)(a1 + 464) = 1;
    *(_BYTE *)(a1 + 401) = v7;
    *(_BYTE *)(a1 + 337) = *(_BYTE *)(a1 + 336);
    *(_BYTE *)(a1 + 241) = *(_BYTE *)(a1 + 240);
    *(_BYTE *)(a1 + 113) = *(_BYTE *)(a1 + 112);
    *(_BYTE *)(a1 + 177) = *(_BYTE *)(a1 + 176);
  }
  if ( *(_QWORD *)(a1 + 304) )
  {
    v9 = (unsigned __int8 *)sub_2674090(*(_QWORD *)(a1 + 296), (__int64)v6);
    v10 = v9;
    if ( *(_BYTE *)(a1 + 113) )
    {
      if ( *(_BYTE *)(a1 + 241) )
      {
LABEL_9:
        v11 = *(unsigned __int8 **)(a1 + 304);
LABEL_10:
        v12 = sub_2674010(v11);
        v13 = sub_ACD640(*((_QWORD *)v12 + 1), *(unsigned __int8 *)(a1 + 464), 0);
        v14 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
        LODWORD(v44) = 1;
        v15 = sub_AAAE30(v14, v13, &v44, 1);
        v16 = *(_QWORD *)(a1 + 304);
        LODWORD(v44) = 0;
        *(_QWORD *)(a1 + 304) = sub_AAAE30(v16, v15, &v44, 1);
        goto LABEL_11;
      }
    }
    else
    {
      v23 = sub_2673FE0(v9);
      v24 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
      LODWORD(v44) = 0;
      v25 = sub_AAAE30(v24, (__int64)v23, &v44, 1);
      v26 = *(_QWORD *)(a1 + 304);
      LODWORD(v44) = 0;
      v27 = sub_AAAE30(v26, v25, &v44, 1);
      v28 = *(_BYTE *)(a1 + 241) == 0;
      *(_QWORD *)(a1 + 304) = v27;
      if ( !v28 )
        goto LABEL_9;
    }
    v18 = sub_2674040(v10);
    v19 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
    LODWORD(v44) = 2;
    v20 = sub_AAAE30(v19, (__int64)v18, &v44, 1);
    v21 = *(_QWORD *)(a1 + 304);
    LODWORD(v44) = 0;
    v22 = sub_AAAE30(v21, v20, &v44, 1);
    *(_QWORD *)(a1 + 304) = v22;
    v11 = (unsigned __int8 *)v22;
    goto LABEL_10;
  }
LABEL_11:
  v51[0] = off_49D3CA8;
  v65[0] = off_4A1FCF8;
  if ( v67 != &v68 )
    _libc_free((unsigned __int64)v67);
  sub_C7D6A0(v65[3], v66, 1);
  v61 = off_4A1FC98;
  if ( v64 != v65 )
    _libc_free((unsigned __int64)v64);
  sub_C7D6A0(v62, 8LL * v63, 8);
  v57[0] = off_4A1FC38;
  if ( v59 != &v60 )
    _libc_free((unsigned __int64)v59);
  sub_C7D6A0(v57[3], 8LL * v58, 8);
  v54[0] = off_4A1FBD8;
  if ( v56 != v57 )
    _libc_free((unsigned __int64)v56);
  sub_C7D6A0(v54[3], 8LL * v55, 8);
  v51[2] = off_4A1FB78;
  if ( v53 != v54 )
    _libc_free((unsigned __int64)v53);
  sub_C7D6A0(v51[5], 8LL * v52, 8);
  return v8;
}
