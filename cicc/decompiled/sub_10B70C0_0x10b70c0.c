// Function: sub_10B70C0
// Address: 0x10b70c0
//
unsigned __int8 *__fastcall sub_10B70C0(__int64 a1, char *a2)
{
  char v4; // al
  char v5; // dl
  unsigned __int8 *v6; // r13
  __int64 v8; // rsi
  __int64 v9; // rcx
  unsigned __int8 *v10; // r14
  __int64 v11; // rax
  char v12; // al
  unsigned __int8 *v13; // r13
  __int64 v14; // r14
  unsigned int v15; // eax
  bool v16; // zf
  int v17; // r13d
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rax
  unsigned __int8 *v30; // r14
  __int64 v31; // rax
  unsigned __int8 *v32; // rdx
  int v33; // eax
  unsigned __int8 *v34; // rdx
  unsigned __int8 *v35; // rcx
  int v36; // eax
  unsigned __int8 *v37; // rcx
  __int64 v38; // rax
  _BYTE *v39; // rdx
  _BYTE *v40; // rdx
  int v41; // ecx
  unsigned __int8 *v42; // rax
  unsigned __int8 *v43; // rax
  _BYTE *v44; // rax
  unsigned __int8 *v45; // rax
  int v46; // ecx
  unsigned __int8 *v47; // rcx
  int v48; // eax
  unsigned __int8 *v49; // rcx
  char v50; // al
  char v51; // al
  char v52; // al
  char v53; // al
  __int64 v54; // rax
  __int64 v55; // [rsp+0h] [rbp-120h]
  unsigned __int8 *v56; // [rsp+8h] [rbp-118h]
  _BYTE *v57; // [rsp+8h] [rbp-118h]
  _BYTE *v58; // [rsp+8h] [rbp-118h]
  _BYTE *v59; // [rsp+8h] [rbp-118h]
  unsigned __int8 *v60; // [rsp+8h] [rbp-118h]
  _BYTE *v61; // [rsp+8h] [rbp-118h]
  unsigned __int8 *v62; // [rsp+8h] [rbp-118h]
  unsigned __int8 *v63; // [rsp+8h] [rbp-118h]
  __int64 v64; // [rsp+10h] [rbp-110h] BYREF
  __int64 v65; // [rsp+18h] [rbp-108h] BYREF
  __int64 v66; // [rsp+20h] [rbp-100h]
  __int64 v67; // [rsp+28h] [rbp-F8h]
  _BYTE v68[32]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v69; // [rsp+50h] [rbp-D0h]
  __int64 *v70; // [rsp+60h] [rbp-C0h] BYREF
  __int64 *v71; // [rsp+68h] [rbp-B8h]
  int v72; // [rsp+70h] [rbp-B0h]
  __int64 v73; // [rsp+78h] [rbp-A8h]
  __int64 v74; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v75; // [rsp+88h] [rbp-98h]
  __int64 v76; // [rsp+90h] [rbp-90h]
  int v77; // [rsp+98h] [rbp-88h]
  __int64 *v78; // [rsp+A0h] [rbp-80h]
  int v79; // [rsp+A8h] [rbp-78h]
  _QWORD v80[2]; // [rsp+B0h] [rbp-70h] BYREF
  int v81; // [rsp+C0h] [rbp-60h]
  __int64 *v82; // [rsp+C8h] [rbp-58h]
  __int64 *v83; // [rsp+D0h] [rbp-50h]
  int v84; // [rsp+D8h] [rbp-48h]
  int v85; // [rsp+E0h] [rbp-40h]
  int v86; // [rsp+E8h] [rbp-38h]

  v4 = *a2;
  v70 = &v64;
  v71 = &v64;
  v72 = 18;
  v5 = v4;
  v73 = (__int64)&v64;
  LODWORD(v75) = 18;
  v76 = (__int64)&v65;
  v77 = 14;
  v78 = &v65;
  v79 = 18;
  LODWORD(v80[0]) = 14;
  v74 = 0x4000000000000000LL;
  if ( v4 != 43 )
    goto LABEL_2;
  v8 = *((_QWORD *)a2 - 8);
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 || *(_QWORD *)(v9 + 8) || *(_BYTE *)v8 != 47 )
    goto LABEL_6;
  v26 = *(_QWORD *)(v8 - 64);
  v10 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  if ( v26 )
  {
    v64 = *(_QWORD *)(v8 - 64);
    v27 = *(_QWORD *)(v8 - 32);
    v28 = *((_QWORD *)v10 + 2);
    if ( v26 == v27 )
    {
      if ( v27 )
      {
        if ( !v28 )
          goto LABEL_2;
        if ( *(_QWORD *)(v28 + 8) )
        {
          v5 = 43;
          goto LABEL_2;
        }
        if ( *v10 != 47 )
          goto LABEL_32;
        v39 = (_BYTE *)*((_QWORD *)v10 - 8);
        if ( *v39 != 43 )
        {
          v40 = (_BYTE *)*((_QWORD *)v10 - 4);
          if ( *v40 != 43 )
            goto LABEL_32;
          v41 = 47;
          goto LABEL_56;
        }
        v44 = (_BYTE *)*((_QWORD *)v39 - 8);
        if ( *v44 == 47 && v26 == *((_QWORD *)v44 - 8) )
        {
          v59 = (_BYTE *)*((_QWORD *)v10 - 8);
          v51 = sub_1009690((double *)&v74, *((_QWORD *)v44 - 4));
          v39 = v59;
          v16 = v51 == 0;
          v45 = (unsigned __int8 *)*((_QWORD *)v59 - 4);
          if ( !v16 && v45 )
            goto LABEL_83;
          v46 = (_DWORD)v75 + 29;
        }
        else
        {
          v45 = (unsigned __int8 *)*((_QWORD *)v39 - 4);
          v46 = 47;
        }
        if ( *v45 != v46
          || *((_QWORD *)v45 - 8) != *(_QWORD *)v73
          || (v61 = v39, !(unsigned __int8)sub_1009690((double *)&v74, *((_QWORD *)v45 - 4)))
          || (v45 = (unsigned __int8 *)*((_QWORD *)v61 - 8)) == 0 )
        {
          v40 = (_BYTE *)*((_QWORD *)v10 - 4);
LABEL_68:
          if ( (unsigned __int8)*v40 != v77 + 29 )
            goto LABEL_6;
          v41 = (_DWORD)v75 + 29;
LABEL_56:
          v42 = (unsigned __int8 *)*((_QWORD *)v40 - 8);
          if ( v41 == *v42 && *((_QWORD *)v42 - 8) == *(_QWORD *)v73 )
          {
            v58 = v40;
            v50 = sub_1009690((double *)&v74, *((_QWORD *)v42 - 4));
            v40 = v58;
            v16 = v50 == 0;
            v43 = (unsigned __int8 *)*((_QWORD *)v58 - 4);
            if ( !v16 && v43 )
            {
LABEL_62:
              *(_QWORD *)v76 = v43;
              if ( *((_QWORD *)v10 - 8) == *v78 )
                goto LABEL_11;
              goto LABEL_6;
            }
            v41 = (_DWORD)v75 + 29;
          }
          else
          {
            v43 = (unsigned __int8 *)*((_QWORD *)v40 - 4);
          }
          if ( *v43 == v41 && *((_QWORD *)v43 - 8) == *(_QWORD *)v73 )
          {
            v57 = v40;
            if ( (unsigned __int8)sub_1009690((double *)&v74, *((_QWORD *)v43 - 4)) )
            {
              v43 = (unsigned __int8 *)*((_QWORD *)v57 - 8);
              if ( v43 )
                goto LABEL_62;
            }
          }
LABEL_6:
          v10 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
          goto LABEL_7;
        }
LABEL_83:
        *(_QWORD *)v76 = v45;
        v40 = (_BYTE *)*((_QWORD *)v10 - 4);
        if ( v40 == (_BYTE *)*v78 )
          goto LABEL_11;
        goto LABEL_68;
      }
    }
  }
LABEL_7:
  v11 = *((_QWORD *)v10 + 2);
  if ( !v11 )
    goto LABEL_8;
  if ( *(_QWORD *)(v11 + 8) )
  {
LABEL_33:
    v5 = *a2;
    goto LABEL_2;
  }
LABEL_32:
  if ( *v10 != v72 + 29 )
    goto LABEL_33;
  v29 = *((_QWORD *)v10 - 8);
  if ( !v29 )
    goto LABEL_33;
  *v70 = v29;
  if ( *((_QWORD *)v10 - 4) != *v71 )
    goto LABEL_8;
  v30 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
  v31 = *((_QWORD *)v30 + 2);
  if ( !v31 || *(_QWORD *)(v31 + 8) || *v30 != v79 + 29 )
    goto LABEL_8;
  v32 = (unsigned __int8 *)*((_QWORD *)v30 - 8);
  v33 = v77 + 29;
  if ( *v32 != v77 + 29 )
  {
    v34 = (unsigned __int8 *)*((_QWORD *)v30 - 4);
    goto LABEL_43;
  }
  v47 = (unsigned __int8 *)*((_QWORD *)v32 - 8);
  v48 = (int)v75;
  if ( *v47 != (_DWORD)v75 + 29 || *((_QWORD *)v47 - 8) != *(_QWORD *)v73 )
    goto LABEL_71;
  v62 = (unsigned __int8 *)*((_QWORD *)v30 - 8);
  v53 = sub_1009690((double *)&v74, *((_QWORD *)v47 - 4));
  v32 = v62;
  if ( !v53 )
  {
    v48 = (int)v75;
LABEL_71:
    v49 = (unsigned __int8 *)*((_QWORD *)v32 - 4);
    goto LABEL_72;
  }
  v49 = (unsigned __int8 *)*((_QWORD *)v62 - 4);
  if ( v49 )
  {
    *(_QWORD *)v76 = v49;
    goto LABEL_99;
  }
  v48 = (int)v75;
LABEL_72:
  if ( *v49 != v48 + 29
    || *((_QWORD *)v49 - 8) != *(_QWORD *)v73
    || (v63 = v32, !(unsigned __int8)sub_1009690((double *)&v74, *((_QWORD *)v49 - 4)))
    || (v54 = *((_QWORD *)v63 - 8)) == 0 )
  {
    v34 = (unsigned __int8 *)*((_QWORD *)v30 - 4);
    v33 = v77 + 29;
    goto LABEL_43;
  }
  *(_QWORD *)v76 = v54;
LABEL_99:
  v34 = (unsigned __int8 *)*((_QWORD *)v30 - 4);
  if ( v34 == (unsigned __int8 *)*v78 )
    goto LABEL_11;
  v33 = v77 + 29;
LABEL_43:
  if ( *v34 != v33 )
    goto LABEL_8;
  v35 = (unsigned __int8 *)*((_QWORD *)v34 - 8);
  v36 = (int)v75;
  if ( *v35 == (_DWORD)v75 + 29 && *((_QWORD *)v35 - 8) == *(_QWORD *)v73 )
  {
    v60 = v34;
    v52 = sub_1009690((double *)&v74, *((_QWORD *)v35 - 4));
    v34 = v60;
    if ( v52 )
    {
      v37 = (unsigned __int8 *)*((_QWORD *)v60 - 4);
      if ( v37 )
      {
        *(_QWORD *)v76 = v37;
LABEL_51:
        if ( *((_QWORD *)v30 - 8) == *v78 )
          goto LABEL_11;
        goto LABEL_8;
      }
      v36 = (int)v75;
      goto LABEL_46;
    }
    v36 = (int)v75;
  }
  v37 = (unsigned __int8 *)*((_QWORD *)v34 - 4);
LABEL_46:
  if ( *v37 == v36 + 29 && *((_QWORD *)v37 - 8) == *(_QWORD *)v73 )
  {
    v56 = v34;
    if ( (unsigned __int8)sub_1009690((double *)&v74, *((_QWORD *)v37 - 4)) )
    {
      v38 = *((_QWORD *)v56 - 8);
      if ( v38 )
      {
        *(_QWORD *)v76 = v38;
        goto LABEL_51;
      }
    }
  }
LABEL_8:
  v5 = *a2;
LABEL_2:
  v70 = &v64;
  v71 = &v65;
  v72 = 18;
  LODWORD(v74) = 18;
  v75 = &v64;
  v77 = 18;
  v78 = &v65;
  v79 = 18;
  v80[0] = &v64;
  v80[1] = &v64;
  v81 = 18;
  v82 = &v65;
  v83 = &v65;
  v84 = 18;
  v85 = 14;
  v86 = 14;
  v73 = 0x4000000000000000LL;
  v76 = 0x4000000000000000LL;
  if ( v5 != 43 )
    return 0;
  v12 = sub_10B5AF0((__int64)&v70, *((unsigned __int8 **)a2 - 8));
  v13 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  if ( (!v12 || !(unsigned __int8)sub_10B0B40((__int64)v80, *((unsigned __int8 **)a2 - 4)))
    && (!(unsigned __int8)sub_10B5AF0((__int64)&v70, v13)
     || !(unsigned __int8)sub_10B0B40((__int64)v80, *((unsigned __int8 **)a2 - 8))) )
  {
    return 0;
  }
LABEL_11:
  v14 = *(_QWORD *)(a1 + 32);
  v69 = 257;
  v15 = sub_B45210((__int64)a2);
  BYTE4(v66) = 1;
  v16 = *(_BYTE *)(v14 + 108) == 0;
  LODWORD(v66) = v15;
  v17 = v15;
  v18 = v65;
  v67 = v66;
  if ( v16 )
  {
    v55 = v64;
    v19 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(v14 + 80) + 40LL))(
            *(_QWORD *)(v14 + 80),
            14,
            v64,
            v65,
            v15);
    if ( !v19 )
    {
      LOWORD(v74) = 257;
      v20 = sub_B504D0(14, v55, v18, (__int64)&v70, 0, 0);
      v21 = *(_QWORD *)(v14 + 96);
      v19 = v20;
      if ( v21 )
        sub_B99FD0(v20, 3u, v21);
      sub_B45150(v19, v17);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v14 + 88) + 16LL))(
        *(_QWORD *)(v14 + 88),
        v19,
        v68,
        *(_QWORD *)(v14 + 56),
        *(_QWORD *)(v14 + 64));
      v22 = *(_QWORD *)v14;
      v23 = *(_QWORD *)v14 + 16LL * *(unsigned int *)(v14 + 8);
      if ( *(_QWORD *)v14 != v23 )
      {
        do
        {
          v24 = *(_QWORD *)(v22 + 8);
          v25 = *(_DWORD *)v22;
          v22 += 16;
          sub_B99FD0(v19, v25, v24);
        }
        while ( v23 != v22 );
      }
    }
  }
  else
  {
    v19 = sub_B35400(v14, 0x66u, v64, v65, v66, (__int64)v68, 0, 0, 0);
  }
  LOWORD(v74) = 257;
  v6 = (unsigned __int8 *)sub_B504D0(18, v19, v19, (__int64)&v70, 0, 0);
  sub_B45260(v6, (__int64)a2, 1);
  return v6;
}
