// Function: sub_2587AA0
// Address: 0x2587aa0
//
__int64 __fastcall sub_2587AA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v7; // rdi
  int v8; // r12d
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  unsigned __int8 *v13; // r15
  __int64 v14; // rsi
  unsigned __int8 *v15; // rdi
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r14
  int v21; // r12d
  unsigned int v22; // r12d
  __int64 v23; // rcx
  unsigned int **v25; // r14
  __int64 v26; // rcx
  __int64 v27; // r15
  unsigned __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // r15
  unsigned int *v31; // r10
  __int64 v32; // rsi
  __int64 v33; // r15
  __int64 (__fastcall *v34)(__int64); // rax
  __int64 (__fastcall *v35)(__int64); // rax
  __int64 v36; // rdx
  char v37; // al
  unsigned int *v38; // rax
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rdi
  __int64 (__fastcall *v43)(__int64); // rax
  __int64 (__fastcall *v45)(__int64); // rax
  __int64 v46; // rdx
  char v47; // al
  __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rax
  char *v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // [rsp-10h] [rbp-F0h]
  __int64 v55; // [rsp+0h] [rbp-E0h]
  __int64 *v56; // [rsp+8h] [rbp-D8h]
  __int64 *v57; // [rsp+28h] [rbp-B8h]
  __int64 v58; // [rsp+28h] [rbp-B8h]
  unsigned int *v59; // [rsp+28h] [rbp-B8h]
  __int64 v60; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v62; // [rsp+40h] [rbp-A0h]
  char *v63; // [rsp+48h] [rbp-98h] BYREF
  __int64 v64; // [rsp+50h] [rbp-90h]
  char v65; // [rsp+58h] [rbp-88h] BYREF
  __int64 v66; // [rsp+60h] [rbp-80h] BYREF
  char *v67; // [rsp+68h] [rbp-78h]
  int v68; // [rsp+70h] [rbp-70h]
  char v69; // [rsp+78h] [rbp-68h] BYREF
  unsigned __int8 *v70; // [rsp+80h] [rbp-60h] BYREF
  unsigned __int64 v71; // [rsp+88h] [rbp-58h] BYREF
  __int64 v72; // [rsp+90h] [rbp-50h] BYREF
  _BYTE v73[72]; // [rsp+98h] [rbp-48h] BYREF

  v7 = *(unsigned __int8 **)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v63 = &v65;
  v64 = 0;
  v62 = v7;
  if ( v8 )
  {
    sub_2538240((__int64)&v63, (char **)(a2 + 8), a3, a4, a5, a6);
    if ( (_DWORD)v64 )
    {
LABEL_25:
      v25 = (unsigned int **)a1[1];
      v71 = (unsigned __int64)v73;
      v72 = 0;
      v70 = v62;
      sub_2538550((__int64)&v71, (__int64)&v63, v9, v23, v10, v11);
      v27 = (__int64)v70;
      v28 = **v25;
      if ( (_DWORD)v72 || sub_B491E0((__int64)v70) )
      {
        v29 = (unsigned int)(v28 + 1);
        v28 = *(unsigned int *)(v71 + 4 * v29);
      }
      v30 = *(_QWORD *)(v27 - 32);
      if ( !v30 || *(_BYTE *)v30 )
        BUG();
      if ( v28 >= *(_QWORD *)(v30 + 104) )
        goto LABEL_41;
      v31 = v25[1];
      if ( (*(_BYTE *)(v30 + 2) & 1) != 0 )
      {
        v59 = v25[1];
        sub_B2C6D0(v30, (__int64)&v63, v29, v26);
        v31 = v59;
      }
      v58 = (__int64)v31;
      sub_250D230((unsigned __int64 *)&v66, *(_QWORD *)(v30 + 96) + 40 * v28, 6, 0);
      v32 = v66;
      v33 = sub_2587260(v58, v66, (__int64)v67, (__int64)v25[2], 0, 0, 1);
      if ( !v33 )
        goto LABEL_41;
      v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v33 + 88) + 16LL);
      v22 = v34 == sub_2505E30
          ? *(unsigned __int8 *)(v33 + 97)
          : ((__int64 (__fastcall *)(__int64, __int64))v34)(v33 + 88, v32);
      if ( !(_BYTE)v22 )
        goto LABEL_41;
      v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v33 + 112LL);
      if ( v35 == sub_2534DF0 )
      {
        v36 = *(_QWORD *)(v33 + 104);
        v37 = *(_BYTE *)(v33 + 112);
      }
      else
      {
        v50 = v35(v33);
        v67 = v51;
        v36 = v50;
        v37 = (char)v67;
        v66 = v36;
      }
      if ( v37 )
      {
        v38 = v25[2];
        if ( !*((_BYTE *)v38 + 112) || *((_QWORD *)v38 + 13) != v36 )
LABEL_41:
          v22 = 0;
      }
      if ( (_BYTE *)v71 != v73 )
        _libc_free(v71);
      goto LABEL_21;
    }
    v7 = v62;
  }
  if ( sub_B491E0((__int64)v7) )
  {
    v23 = (unsigned int)v64;
    if ( !(_DWORD)v64 )
    {
      v22 = 0;
      goto LABEL_21;
    }
    goto LABEL_25;
  }
  v12 = *a1;
  v70 = (unsigned __int8 *)&v72;
  v71 = 0x400000000LL;
  sub_E33A00(v62, (__int64)&v70);
  v13 = &v70[8 * (unsigned int)v71];
  if ( v70 == v13 )
  {
    v22 = 1;
  }
  else
  {
    v57 = (__int64 *)v70;
    v56 = (__int64 *)&v70[8 * (unsigned int)v71];
    while ( 1 )
    {
      v14 = *v57;
      sub_E33C60(&v66, *v57);
      if ( v68 || sub_B491E0(v66) )
      {
        v15 = *(unsigned __int8 **)(v66
                                  + 32 * (*(unsigned int *)v67 - (unsigned __int64)(*(_DWORD *)(v66 + 4) & 0x7FFFFFF)));
        if ( !v15 )
          goto LABEL_75;
      }
      else
      {
        v15 = *(unsigned __int8 **)(v66 - 32);
        if ( !v15 )
LABEL_75:
          BUG();
      }
      v16 = sub_BD3990(v15, v14);
      if ( *v16 )
        goto LABEL_75;
      if ( (v16[2] & 1) != 0 )
      {
        v55 = (__int64)v16;
        sub_B2C6D0((__int64)v16, v14, v17, v18);
        v19 = *(_QWORD *)(v55 + 96);
        v20 = v19 + 40LL * *(_QWORD *)(v55 + 104);
        if ( (*(_BYTE *)(v55 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v55, v14, 5LL * *(_QWORD *)(v55 + 104), v49);
          v19 = *(_QWORD *)(v55 + 96);
        }
      }
      else
      {
        v19 = *((_QWORD *)v16 + 12);
        v20 = v19 + 40LL * *((_QWORD *)v16 + 13);
      }
      if ( v20 != v19 )
        break;
LABEL_15:
      if ( v67 != &v69 )
        _libc_free((unsigned __int64)v67);
      if ( v56 == ++v57 )
      {
        v13 = v70;
        v22 = 1;
        goto LABEL_58;
      }
    }
    while ( 1 )
    {
      v21 = *(_DWORD *)(v19 + 32);
      if ( v68 || sub_B491E0(v66) )
        v21 = *(_DWORD *)&v67[4 * (v21 + 1)];
      if ( **(_DWORD **)v12 == v21 )
      {
        v39 = *(_QWORD *)(v12 + 8);
        sub_250D230((unsigned __int64 *)&v60, v19, 6, 0);
        v40 = sub_2587260(v39, v60, v61, *(_QWORD *)(v12 + 16), 0, 0, 1);
        v41 = v40;
        if ( !v40 )
          break;
        v42 = v40 + 88;
        v43 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v40 + 88) + 16LL);
        if ( !(v43 == sub_2505E30 ? *(_BYTE *)(v41 + 97) : ((__int64 (__fastcall *)(__int64, __int64))v43)(v42, v54)) )
          break;
        v45 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v41 + 112LL);
        if ( v45 == sub_2534DF0 )
        {
          v46 = *(_QWORD *)(v41 + 104);
          v47 = *(_BYTE *)(v41 + 112);
        }
        else
        {
          v52 = ((__int64 (__fastcall *)(__int64, __int64))v45)(v41, v54);
          v61 = v53;
          v46 = v52;
          v47 = v61;
          v60 = v46;
        }
        if ( v47 )
        {
          v48 = *(_QWORD *)(v12 + 16);
          if ( !*(_BYTE *)(v48 + 112) || *(_QWORD *)(v48 + 104) != v46 )
            break;
        }
      }
      v19 += 40LL;
      if ( v19 == v20 )
        goto LABEL_15;
    }
    v22 = 0;
    if ( v67 != &v69 )
      _libc_free((unsigned __int64)v67);
    v13 = v70;
  }
LABEL_58:
  if ( v13 != (unsigned __int8 *)&v72 )
    _libc_free((unsigned __int64)v13);
LABEL_21:
  if ( v63 != &v65 )
    _libc_free((unsigned __int64)v63);
  return v22;
}
