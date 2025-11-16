// Function: sub_2D674A0
// Address: 0x2d674a0
//
__int64 __fastcall sub_2D674A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  char *v6; // r13
  char v10; // al
  char *v11; // rcx
  char *v12; // rdx
  char v13; // al
  char *v14; // r8
  unsigned __int64 v15; // r13
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char v19; // bl
  _BYTE *v20; // rdi
  _BYTE *v21; // r13
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  void (__fastcall *v28)(_BYTE *, __int64, __int64); // rax
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 *v37; // rdx
  char v38; // al
  __int64 *v39; // [rsp-10h] [rbp-420h]
  _BYTE *v40; // [rsp+20h] [rbp-3F0h]
  __int64 v41; // [rsp+48h] [rbp-3C8h]
  __int64 v42; // [rsp+50h] [rbp-3C0h]
  __int64 v43; // [rsp+58h] [rbp-3B8h]
  unsigned int v44; // [rsp+60h] [rbp-3B0h]
  char v45; // [rsp+67h] [rbp-3A9h]
  __int64 v46; // [rsp+68h] [rbp-3A8h]
  __int64 v47; // [rsp+70h] [rbp-3A0h]
  __int64 v48; // [rsp+78h] [rbp-398h]
  __int64 v49; // [rsp+80h] [rbp-390h]
  char *v50; // [rsp+88h] [rbp-388h]
  __int64 v51; // [rsp+88h] [rbp-388h]
  _QWORD v52[2]; // [rsp+90h] [rbp-380h] BYREF
  __int64 v53; // [rsp+A0h] [rbp-370h]
  __int64 v54; // [rsp+A8h] [rbp-368h]
  _BYTE v55[16]; // [rsp+B0h] [rbp-360h] BYREF
  void (__fastcall *v56)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-350h]
  __int64 v57; // [rsp+C8h] [rbp-348h]
  _OWORD v58[4]; // [rsp+D0h] [rbp-340h] BYREF
  __int64 v59; // [rsp+110h] [rbp-300h]
  _QWORD v60[5]; // [rsp+120h] [rbp-2F0h] BYREF
  _BYTE v61[16]; // [rsp+148h] [rbp-2C8h] BYREF
  void (__fastcall *v62)(_BYTE *, _BYTE *, __int64); // [rsp+158h] [rbp-2B8h]
  __int64 v63; // [rsp+160h] [rbp-2B0h]
  __int64 v64; // [rsp+168h] [rbp-2A8h]
  unsigned int v65; // [rsp+170h] [rbp-2A0h]
  __int64 v66; // [rsp+178h] [rbp-298h]
  _OWORD *v67; // [rsp+180h] [rbp-290h]
  __int64 v68; // [rsp+188h] [rbp-288h]
  __int64 v69; // [rsp+190h] [rbp-280h]
  __int64 v70; // [rsp+198h] [rbp-278h]
  _QWORD *v71; // [rsp+1A0h] [rbp-270h]
  char v72; // [rsp+1A8h] [rbp-268h]
  char v73; // [rsp+1A9h] [rbp-267h]
  __int64 v74; // [rsp+1B0h] [rbp-260h]
  __int64 v75; // [rsp+1B8h] [rbp-258h]
  _BYTE *v76; // [rsp+1C0h] [rbp-250h] BYREF
  __int64 v77; // [rsp+1C8h] [rbp-248h]
  _BYTE v78[256]; // [rsp+1D0h] [rbp-240h] BYREF
  __int64 *v79; // [rsp+2D0h] [rbp-140h] BYREF
  unsigned __int64 v80; // [rsp+2D8h] [rbp-138h]
  __int64 v81; // [rsp+2E0h] [rbp-130h] BYREF
  int v82; // [rsp+2E8h] [rbp-128h]
  char v83; // [rsp+2ECh] [rbp-124h]
  char v84; // [rsp+2F0h] [rbp-120h] BYREF

  v4 = *(unsigned __int8 *)(a1 + 136);
  if ( (_BYTE)v4 )
    return 1;
  v6 = *(char **)(a4 + 48);
  v50 = *(char **)(a4 + 40);
  v10 = sub_2D56CB0(a1, v50, *(char **)(a3 + 40), *(char **)(a3 + 48));
  v11 = *(char **)(a3 + 48);
  v12 = *(char **)(a3 + 40);
  if ( v10 )
  {
    v38 = sub_2D56CB0(a1, v6, v12, v11);
    v14 = v6;
    if ( v38 )
      return 1;
  }
  else
  {
    v13 = sub_2D56CB0(a1, v6, v12, v11);
    v14 = v50;
    v15 = (unsigned __int64)v50 | (unsigned __int64)v6;
    if ( !v13 )
      v14 = (char *)v15;
  }
  if ( !v14 )
    return 1;
  v16 = *(_QWORD *)(a1 + 8);
  v76 = v78;
  v77 = 0x1000000000LL;
  v80 = (unsigned __int64)&v84;
  v39 = *(__int64 **)(a1 + 152);
  v17 = *(_QWORD *)(a1 + 16);
  v18 = *(_QWORD *)(a1 + 144);
  LODWORD(v60[0]) = 0;
  v79 = 0;
  v81 = 16;
  v82 = 0;
  v83 = 1;
  v19 = sub_2D5DA80(a2, (__int64)&v76, (__int64)&v79, v16, v17, v18, v39, v60);
  if ( !v83 )
    _libc_free(v80);
  v20 = v76;
  if ( v19 )
    goto LABEL_40;
  v79 = &v81;
  v80 = 0x2000000000LL;
  v40 = &v76[16 * (unsigned int)v77];
  if ( v40 == v76 )
  {
    v4 = 1;
    goto LABEL_40;
  }
  v21 = v76;
  while ( 1 )
  {
    v22 = **(_QWORD **)v21;
    v51 = *(_QWORD *)(*(_QWORD *)v21 + 24LL);
    v41 = *((_QWORD *)v21 + 1);
    v23 = *(_QWORD *)(v22 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 <= 1 )
      v23 = **(_QWORD **)(v23 + 16);
    v24 = *(_DWORD *)(v23 + 8);
    v25 = *(_QWORD *)(a1 + 120);
    v59 = 1;
    v52[1] = 0;
    v44 = v24 >> 8;
    v53 = 0;
    v54 = 0;
    memset(v58, 0, sizeof(v58));
    v26 = *(unsigned int *)(v25 + 8);
    v52[0] = 0;
    v49 = 0;
    if ( (_DWORD)v26 )
      v49 = *(_QWORD *)(*(_QWORD *)v25 + 8 * v26 - 8);
    v27 = *(_QWORD *)(a1 + 152);
    v56 = 0;
    v47 = v27;
    v46 = *(_QWORD *)(a1 + 144);
    v45 = *(_BYTE *)(a1 + 137);
    v43 = *(_QWORD *)(a1 + 112);
    v42 = *(_QWORD *)(a1 + 104);
    v28 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a1 + 56);
    if ( v28 )
    {
      v28(v55, a1 + 40, 2);
      v57 = *(_QWORD *)(a1 + 64);
      v56 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a1 + 56);
    }
    v29 = *(_QWORD *)(a1 + 16);
    v30 = *(_QWORD *)(a1 + 8);
    v48 = *(_QWORD *)(a1 + 32);
    v60[0] = &v79;
    v60[1] = v30;
    v60[2] = v29;
    v31 = sub_B43CC0(v51);
    v62 = 0;
    v60[3] = v31;
    v60[4] = v48;
    if ( v56 )
    {
      v56(v61, v55, 2);
      v70 = v25;
      v72 = 0;
      v66 = v51;
      v71 = v52;
      v67 = v58;
      v73 = v45;
      v63 = v57;
      v64 = v41;
      v68 = v42;
      v74 = v46;
      v62 = v56;
      v65 = v44;
      v69 = v43;
      v75 = v47;
      if ( v56 )
        v56(v55, v55, 3);
    }
    else
    {
      v70 = v25;
      v64 = v41;
      v65 = v44;
      v66 = v51;
      v67 = v58;
      v68 = v42;
      v69 = v43;
      v71 = v52;
      v73 = v45;
      v74 = v46;
      v75 = v47;
    }
    v72 = 1;
    sub_2D65BF0((__int64)v60, (unsigned __int8 *)v22, 0);
    sub_2D57BD0(*(__int64 **)(a1 + 120), v49);
    v32 = v79;
    v33 = 8LL * (unsigned int)v80;
    v34 = &v79[(unsigned __int64)v33 / 8];
    v35 = v33 >> 3;
    v36 = v33 >> 5;
    if ( v36 )
    {
      v37 = &v79[4 * v36];
      while ( a2 != *v32 )
      {
        if ( a2 == v32[1] )
        {
          ++v32;
          goto LABEL_29;
        }
        if ( a2 == v32[2] )
        {
          v32 += 2;
          goto LABEL_29;
        }
        if ( a2 == v32[3] )
        {
          v32 += 3;
          goto LABEL_29;
        }
        v32 += 4;
        if ( v37 == v32 )
        {
          v35 = v34 - v32;
          goto LABEL_46;
        }
      }
      goto LABEL_29;
    }
LABEL_46:
    if ( v35 == 2 )
      goto LABEL_50;
    if ( v35 == 3 )
    {
      if ( a2 == *v32 )
        goto LABEL_29;
      ++v32;
LABEL_50:
      if ( a2 == *v32 )
        goto LABEL_29;
      ++v32;
      goto LABEL_52;
    }
    if ( v35 != 1 )
      break;
LABEL_52:
    if ( a2 != *v32 )
      break;
LABEL_29:
    if ( v34 == v32 )
      break;
    LODWORD(v80) = 0;
    if ( v62 )
      v62(v61, v61, 3);
    if ( v53 != -4096 && v53 != 0 && v53 != -8192 )
      sub_BD60C0(v52);
    v21 += 16;
    if ( v40 == v21 )
    {
      v4 = 1;
      goto LABEL_37;
    }
  }
  v4 = 0;
  sub_A17130((__int64)v61);
  sub_D68D70(v52);
LABEL_37:
  if ( v79 != &v81 )
    _libc_free((unsigned __int64)v79);
  v20 = v76;
LABEL_40:
  if ( v20 != v78 )
    _libc_free((unsigned __int64)v20);
  return v4;
}
