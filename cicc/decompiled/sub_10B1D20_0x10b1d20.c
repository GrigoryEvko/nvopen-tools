// Function: sub_10B1D20
// Address: 0x10b1d20
//
__int64 __fastcall sub_10B1D20(__int64 a1, char *a2)
{
  char v4; // al
  char v5; // dl
  __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned __int8 *v9; // r15
  __int64 v10; // rax
  bool v11; // al
  unsigned __int8 *v12; // r15
  __int64 *v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int8 *v25; // r15
  __int64 v26; // rax
  unsigned __int8 *v27; // rdx
  int v28; // esi
  unsigned __int8 *v29; // rdx
  _BYTE *v30; // rdx
  _BYTE *v31; // rdx
  int v32; // ecx
  unsigned __int8 *v33; // rax
  unsigned __int8 *v34; // rax
  _BYTE *v35; // rax
  unsigned __int8 *v36; // rax
  int v37; // ecx
  unsigned __int8 *v38; // rax
  int v39; // ecx
  unsigned __int8 *v40; // rax
  bool v41; // al
  bool v42; // zf
  bool v43; // al
  bool v44; // al
  _BYTE *v45; // [rsp+8h] [rbp-108h]
  _BYTE *v46; // [rsp+8h] [rbp-108h]
  _BYTE *v47; // [rsp+8h] [rbp-108h]
  _BYTE *v48; // [rsp+8h] [rbp-108h]
  unsigned __int8 *v49; // [rsp+8h] [rbp-108h]
  unsigned __int8 *v50; // [rsp+8h] [rbp-108h]
  __int64 v51; // [rsp+10h] [rbp-100h] BYREF
  __int64 v52; // [rsp+18h] [rbp-F8h] BYREF
  char v53[32]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v54; // [rsp+40h] [rbp-D0h]
  __int64 *v55; // [rsp+50h] [rbp-C0h] BYREF
  __int64 *v56; // [rsp+58h] [rbp-B8h]
  int v57; // [rsp+60h] [rbp-B0h]
  __int64 v58; // [rsp+68h] [rbp-A8h] BYREF
  __int64 v59; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v60; // [rsp+78h] [rbp-98h]
  __int64 v61; // [rsp+80h] [rbp-90h]
  int v62; // [rsp+88h] [rbp-88h]
  __int64 *v63; // [rsp+90h] [rbp-80h]
  int v64; // [rsp+98h] [rbp-78h]
  _QWORD v65[2]; // [rsp+A0h] [rbp-70h] BYREF
  int v66; // [rsp+B0h] [rbp-60h]
  __int64 *v67; // [rsp+B8h] [rbp-58h]
  __int64 *v68; // [rsp+C0h] [rbp-50h]
  int v69; // [rsp+C8h] [rbp-48h]
  int v70; // [rsp+D0h] [rbp-40h]
  int v71; // [rsp+D8h] [rbp-38h]

  v4 = *a2;
  v55 = &v51;
  v56 = &v51;
  v57 = 17;
  v5 = v4;
  v58 = (__int64)&v51;
  v59 = 1;
  LODWORD(v60) = 25;
  v61 = (__int64)&v52;
  v62 = 13;
  v63 = &v52;
  v64 = 17;
  LODWORD(v65[0]) = 13;
  if ( v4 != 42 )
    goto LABEL_2;
  v7 = *((_QWORD *)a2 - 8);
  v8 = *(_QWORD *)(v7 + 16);
  if ( !v8 || *(_QWORD *)(v8 + 8) || *(_BYTE *)v7 != 46 )
    goto LABEL_5;
  v21 = *(_QWORD *)(v7 - 64);
  v9 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  if ( v21 )
  {
    v51 = *(_QWORD *)(v7 - 64);
    v22 = *(_QWORD *)(v7 - 32);
    v23 = *((_QWORD *)v9 + 2);
    if ( v22 )
    {
      if ( v21 == v22 )
      {
        if ( !v23 )
          goto LABEL_2;
        if ( *(_QWORD *)(v23 + 8) )
        {
          v5 = 42;
          goto LABEL_2;
        }
        if ( *v9 != 46 )
          goto LABEL_27;
        v30 = (_BYTE *)*((_QWORD *)v9 - 8);
        if ( *v30 != 42 )
        {
          v31 = (_BYTE *)*((_QWORD *)v9 - 4);
          if ( *v31 != 42 )
            goto LABEL_27;
          v32 = 54;
          goto LABEL_44;
        }
        v35 = (_BYTE *)*((_QWORD *)v30 - 8);
        if ( *v35 == 54 && v21 == *((_QWORD *)v35 - 8) )
        {
          v46 = (_BYTE *)*((_QWORD *)v9 - 8);
          v41 = sub_F17ED0(&v59, *((_QWORD *)v35 - 4));
          v30 = v46;
          v42 = !v41;
          v36 = (unsigned __int8 *)*((_QWORD *)v46 - 4);
          if ( !v42 && v36 )
            goto LABEL_66;
          v37 = (_DWORD)v60 + 29;
        }
        else
        {
          v36 = (unsigned __int8 *)*((_QWORD *)v30 - 4);
          v37 = 54;
        }
        if ( v37 != *v36
          || *((_QWORD *)v36 - 8) != *(_QWORD *)v58
          || (v48 = v30, !sub_F17ED0(&v59, *((_QWORD *)v36 - 4)))
          || (v36 = (unsigned __int8 *)*((_QWORD *)v48 - 8)) == 0 )
        {
          v31 = (_BYTE *)*((_QWORD *)v9 - 4);
LABEL_56:
          if ( (unsigned __int8)*v31 != v62 + 29 )
            goto LABEL_5;
          v32 = (_DWORD)v60 + 29;
LABEL_44:
          v33 = (unsigned __int8 *)*((_QWORD *)v31 - 8);
          if ( *v33 == v32 && *((_QWORD *)v33 - 8) == *(_QWORD *)v58 )
          {
            v47 = v31;
            v43 = sub_F17ED0(&v59, *((_QWORD *)v33 - 4));
            v31 = v47;
            v42 = !v43;
            v34 = (unsigned __int8 *)*((_QWORD *)v47 - 4);
            if ( !v42 && v34 )
            {
LABEL_50:
              *(_QWORD *)v61 = v34;
              if ( *((_QWORD *)v9 - 8) == *v63 )
                goto LABEL_10;
              goto LABEL_5;
            }
            v32 = (_DWORD)v60 + 29;
          }
          else
          {
            v34 = (unsigned __int8 *)*((_QWORD *)v31 - 4);
          }
          if ( *v34 == v32 && *((_QWORD *)v34 - 8) == *(_QWORD *)v58 )
          {
            v45 = v31;
            if ( sub_F17ED0(&v59, *((_QWORD *)v34 - 4)) )
            {
              v34 = (unsigned __int8 *)*((_QWORD *)v45 - 8);
              if ( v34 )
                goto LABEL_50;
            }
          }
LABEL_5:
          v9 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
          goto LABEL_6;
        }
LABEL_66:
        *(_QWORD *)v61 = v36;
        v31 = (_BYTE *)*((_QWORD *)v9 - 4);
        if ( v31 == (_BYTE *)*v63 )
          goto LABEL_10;
        goto LABEL_56;
      }
    }
  }
LABEL_6:
  v10 = *((_QWORD *)v9 + 2);
  if ( !v10 )
    goto LABEL_7;
  if ( *(_QWORD *)(v10 + 8) )
  {
LABEL_28:
    v5 = *a2;
    goto LABEL_2;
  }
LABEL_27:
  if ( *v9 != v57 + 29 )
    goto LABEL_28;
  v24 = *((_QWORD *)v9 - 8);
  if ( !v24 )
    goto LABEL_28;
  *v55 = v24;
  if ( *((_QWORD *)v9 - 4) == *v56 )
  {
    v25 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    v26 = *((_QWORD *)v25 + 2);
    if ( v26 )
    {
      if ( !*(_QWORD *)(v26 + 8) && *v25 == v64 + 29 )
      {
        v27 = (unsigned __int8 *)*((_QWORD *)v25 - 8);
        v28 = v62;
        if ( *v27 != v62 + 29 )
        {
          v29 = (unsigned __int8 *)*((_QWORD *)v25 - 4);
LABEL_38:
          if ( sub_10B1B90((__int64)&v58, v28, v29) && *((_QWORD *)v25 - 8) == *v63 )
            goto LABEL_10;
          goto LABEL_7;
        }
        v38 = (unsigned __int8 *)*((_QWORD *)v27 - 8);
        v39 = (int)v60;
        if ( *v38 == (_DWORD)v60 + 29 && *((_QWORD *)v38 - 8) == *(_QWORD *)v58 )
        {
          v49 = (unsigned __int8 *)*((_QWORD *)v25 - 8);
          v44 = sub_F17ED0(&v59, *((_QWORD *)v38 - 4));
          v27 = v49;
          if ( v44 )
          {
            v40 = (unsigned __int8 *)*((_QWORD *)v49 - 4);
            if ( v40 )
              goto LABEL_81;
            v39 = (int)v60;
LABEL_60:
            if ( *v40 != v39 + 29
              || *((_QWORD *)v40 - 8) != *(_QWORD *)v58
              || (v50 = v27, !sub_F17ED0(&v59, *((_QWORD *)v40 - 4)))
              || (v40 = (unsigned __int8 *)*((_QWORD *)v50 - 8)) == 0 )
            {
              v29 = (unsigned __int8 *)*((_QWORD *)v25 - 4);
              v28 = v62;
              goto LABEL_38;
            }
LABEL_81:
            *(_QWORD *)v61 = v40;
            v29 = (unsigned __int8 *)*((_QWORD *)v25 - 4);
            if ( v29 == (unsigned __int8 *)*v63 )
              goto LABEL_10;
            v28 = v62;
            goto LABEL_38;
          }
          v39 = (int)v60;
        }
        v40 = (unsigned __int8 *)*((_QWORD *)v27 - 4);
        goto LABEL_60;
      }
    }
  }
LABEL_7:
  v5 = *a2;
LABEL_2:
  v55 = &v51;
  v56 = &v52;
  v57 = 17;
  v58 = 1;
  LODWORD(v59) = 25;
  v60 = &v51;
  v61 = 1;
  v62 = 25;
  v63 = &v52;
  v64 = 17;
  v65[0] = &v51;
  v65[1] = &v51;
  v66 = 17;
  v67 = &v52;
  v68 = &v52;
  v69 = 17;
  v70 = 13;
  v71 = 13;
  if ( v5 != 42 )
    return 0;
  v11 = sub_10B09E0((__int64)&v55, *((unsigned __int8 **)a2 - 8));
  v12 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  if ( (!v11 || !(unsigned __int8)sub_10B0B40((__int64)v65, *((unsigned __int8 **)a2 - 4)))
    && (!sub_10B09E0((__int64)&v55, v12) || !(unsigned __int8)sub_10B0B40((__int64)v65, *((unsigned __int8 **)a2 - 8))) )
  {
    return 0;
  }
LABEL_10:
  v13 = *(__int64 **)(a1 + 32);
  v54 = 257;
  v14 = v52;
  v15 = v51;
  v16 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v13[10] + 32LL))(
          v13[10],
          13,
          v51,
          v52,
          0,
          0);
  if ( !v16 )
  {
    LOWORD(v59) = 257;
    v16 = sub_B504D0(13, v15, v14, (__int64)&v55, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v13[11] + 16LL))(
      v13[11],
      v16,
      v53,
      v13[7],
      v13[8]);
    v17 = *v13;
    v18 = *v13 + 16LL * *((unsigned int *)v13 + 2);
    while ( v18 != v17 )
    {
      v19 = *(_QWORD *)(v17 + 8);
      v20 = *(_DWORD *)v17;
      v17 += 16;
      sub_B99FD0(v16, v20, v19);
    }
  }
  LOWORD(v59) = 257;
  return sub_B504D0(17, v16, v16, (__int64)&v55, 0, 0);
}
