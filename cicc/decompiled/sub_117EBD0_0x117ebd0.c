// Function: sub_117EBD0
// Address: 0x117ebd0
//
__int64 __fastcall sub_117EBD0(unsigned __int8 *a1, __int64 a2)
{
  _BYTE *v2; // r9
  __int64 v3; // r13
  __int64 v4; // r14
  _BYTE *v6; // rbx
  __int64 v7; // rcx
  unsigned __int8 *v8; // r12
  unsigned int v9; // r15d
  int v10; // eax
  bool v11; // al
  int v12; // eax
  __int64 v13; // rax
  _BYTE *v14; // rax
  const void **v15; // r15
  unsigned int v16; // ebx
  unsigned __int64 v17; // rax
  const void **v18; // rsi
  unsigned __int64 v19; // rax
  const void *v20; // r15
  const void **v21; // rdi
  unsigned int v22; // edx
  bool v23; // al
  bool v24; // al
  const void **v25; // rsi
  __int64 v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // r15
  __int64 v29; // rdx
  _BYTE *v30; // rax
  unsigned __int8 *v31; // rcx
  unsigned int v32; // r15d
  int v33; // eax
  int v34; // ebx
  const void *v35; // rcx
  bool v36; // r13
  unsigned int v37; // r12d
  unsigned __int8 *v38; // rbx
  __int64 v39; // rax
  unsigned int v40; // r13d
  bool v41; // r15
  __int64 v42; // r14
  __int64 *v43; // rdx
  _BYTE *v44; // r14
  _BYTE *v45; // rax
  unsigned int *v46; // rbx
  __int64 v47; // r13
  __int64 v48; // rdx
  unsigned int v49; // esi
  unsigned int v50; // [rsp+Ch] [rbp-104h]
  unsigned int v51; // [rsp+10h] [rbp-100h]
  const void **v52; // [rsp+10h] [rbp-100h]
  __int64 v53; // [rsp+10h] [rbp-100h]
  unsigned int v54; // [rsp+10h] [rbp-100h]
  __int64 v55; // [rsp+18h] [rbp-F8h]
  _BYTE *v56; // [rsp+18h] [rbp-F8h]
  __int64 v57; // [rsp+18h] [rbp-F8h]
  const void *v58; // [rsp+20h] [rbp-F0h]
  __int64 v59; // [rsp+20h] [rbp-F0h]
  unsigned int v60; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v61; // [rsp+28h] [rbp-E8h]
  _BYTE *v62; // [rsp+28h] [rbp-E8h]
  _BYTE *v63; // [rsp+30h] [rbp-E0h]
  _BYTE *v64; // [rsp+30h] [rbp-E0h]
  int v65; // [rsp+30h] [rbp-E0h]
  int v66; // [rsp+30h] [rbp-E0h]
  const void **v68; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v69; // [rsp+50h] [rbp-C0h] BYREF
  const void **v70; // [rsp+58h] [rbp-B8h] BYREF
  unsigned __int64 v71; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v72; // [rsp+68h] [rbp-A8h]
  const void *v73; // [rsp+70h] [rbp-A0h] BYREF
  int v74; // [rsp+78h] [rbp-98h]
  const char *v75; // [rsp+80h] [rbp-90h] BYREF
  __int64 *v76; // [rsp+88h] [rbp-88h] BYREF
  const char *v77; // [rsp+90h] [rbp-80h]
  const void ***v78; // [rsp+98h] [rbp-78h] BYREF
  __int16 v79; // [rsp+A0h] [rbp-70h]
  const void *v80; // [rsp+B0h] [rbp-60h] BYREF
  const void ***v81; // [rsp+B8h] [rbp-58h] BYREF
  char v82; // [rsp+C0h] [rbp-50h]
  __int64 *v83; // [rsp+C8h] [rbp-48h] BYREF
  __int16 v84; // [rsp+D0h] [rbp-40h]

  v2 = (_BYTE *)*((_QWORD *)a1 - 12);
  v3 = *((_QWORD *)a1 - 8);
  v4 = *((_QWORD *)a1 - 4);
  if ( *v2 != 82 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)v2 - 8);
  if ( !v6 )
    return 0;
  v7 = *((_QWORD *)v2 - 4);
  v8 = a1;
  if ( *(_BYTE *)v7 == 17 )
  {
    v9 = *(_DWORD *)(v7 + 32);
    if ( v9 <= 0x40 )
    {
      v11 = *(_QWORD *)(v7 + 24) == 0;
    }
    else
    {
      v63 = (_BYTE *)*((_QWORD *)a1 - 12);
      v10 = sub_C444A0(v7 + 24);
      v2 = v63;
      v11 = v9 == v10;
    }
  }
  else
  {
    v28 = *(_QWORD *)(v7 + 8);
    v64 = (_BYTE *)*((_QWORD *)a1 - 12);
    v29 = (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17;
    if ( (unsigned int)v29 > 1 || *(_BYTE *)v7 > 0x15u )
      return 0;
    v61 = (unsigned __int8 *)*((_QWORD *)v2 - 4);
    v30 = sub_AD7630(v7, 0, v29);
    v31 = v61;
    v2 = v64;
    if ( !v30 || *v30 != 17 )
    {
      if ( *(_BYTE *)(v28 + 8) == 17 )
      {
        v66 = *(_DWORD *)(v28 + 32);
        if ( v66 )
        {
          v62 = v2;
          v59 = v3;
          v36 = 0;
          v37 = 0;
          v56 = v6;
          v38 = v31;
          while ( 1 )
          {
            v39 = sub_AD69F0(v38, v37);
            if ( !v39 )
              break;
            if ( *(_BYTE *)v39 != 13 )
            {
              if ( *(_BYTE *)v39 != 17 )
                break;
              v40 = *(_DWORD *)(v39 + 32);
              v36 = v40 <= 0x40 ? *(_QWORD *)(v39 + 24) == 0 : v40 == (unsigned int)sub_C444A0(v39 + 24);
              if ( !v36 )
                break;
            }
            if ( v66 == ++v37 )
            {
              v8 = a1;
              v41 = v36;
              v2 = v62;
              v6 = v56;
              v3 = v59;
              if ( v41 )
                goto LABEL_9;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v32 = *((_DWORD *)v30 + 8);
    if ( v32 <= 0x40 )
    {
      v11 = *((_QWORD *)v30 + 3) == 0;
    }
    else
    {
      v33 = sub_C444A0((__int64)(v30 + 24));
      v2 = v64;
      v11 = v32 == v33;
    }
  }
  if ( !v11 )
    return 0;
LABEL_9:
  v12 = sub_B53900((__int64)v2);
  if ( (unsigned int)(v12 - 32) > 1 )
    return 0;
  if ( v12 == 33 )
  {
    v13 = v3;
    v3 = v4;
    v4 = v13;
  }
  v80 = (const void *)v3;
  v81 = &v68;
  v82 = 1;
  if ( *v6 != 57 || v3 != *((_QWORD *)v6 - 8) || !(unsigned __int8)sub_991580((__int64)&v81, *((_QWORD *)v6 - 4)) )
    return 0;
  v75 = (const char *)v3;
  v76 = &v69;
  LOBYTE(v77) = 1;
  v78 = &v70;
  LOBYTE(v79) = 1;
  if ( *(_BYTE *)v4 != 57
    || (v27 = *(_BYTE **)(v4 - 64), *v27 != 42)
    || v3 != *((_QWORD *)v27 - 8)
    || !(unsigned __int8)sub_991580((__int64)&v76, *((_QWORD *)v27 - 4))
    || !(unsigned __int8)sub_991580((__int64)&v78, *(_QWORD *)(v4 - 32)) )
  {
    v80 = (const void *)v3;
    v81 = &v70;
    v82 = 1;
    v83 = &v69;
    LOBYTE(v84) = 1;
    if ( *(_BYTE *)v4 != 42 )
      return 0;
    v14 = *(_BYTE **)(v4 - 64);
    if ( *v14 != 57
      || v3 != *((_QWORD *)v14 - 8)
      || !(unsigned __int8)sub_991580((__int64)&v81, *((_QWORD *)v14 - 4))
      || !(unsigned __int8)sub_991580((__int64)&v83, *(_QWORD *)(v4 - 32)) )
    {
      return 0;
    }
  }
  v15 = v68;
  if ( *((_DWORD *)v68 + 2) > 0x40u )
  {
    v65 = *((_DWORD *)v68 + 2);
    v34 = sub_C445E0((__int64)v68);
    if ( !v34 || v65 != (unsigned int)sub_C444A0((__int64)v15) + v34 )
      return 0;
  }
  else if ( !*v68 || ((unsigned __int64)*v68 & ((unsigned __int64)*v68 + 1)) != 0 )
  {
    return 0;
  }
  v16 = *((_DWORD *)v15 + 2);
  LODWORD(v81) = v16;
  if ( v16 <= 0x40 )
  {
    v17 = (unsigned __int64)*v15;
LABEL_26:
    v72 = v16;
    v18 = v70;
    v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v17;
    if ( !v16 )
      v19 = 0;
    v71 = v19;
LABEL_29:
    v20 = *v18;
    if ( (const void *)v19 == *v18 )
      goto LABEL_30;
    return 0;
  }
  sub_C43780((__int64)&v80, v15);
  v16 = (unsigned int)v81;
  if ( (unsigned int)v81 <= 0x40 )
  {
    v17 = (unsigned __int64)v80;
    goto LABEL_26;
  }
  sub_C43D10((__int64)&v80);
  v16 = (unsigned int)v81;
  v20 = v80;
  v18 = v70;
  v72 = (unsigned int)v81;
  v71 = (unsigned __int64)v80;
  if ( (unsigned int)v81 <= 0x40 )
  {
    v19 = (unsigned __int64)v80;
    goto LABEL_29;
  }
  if ( !sub_C43C50((__int64)&v71, v70) )
  {
    v4 = 0;
LABEL_43:
    if ( v20 )
      j_j___libc_free_0_0(v20);
    return v4;
  }
LABEL_30:
  LODWORD(v81) = *((_DWORD *)v68 + 2);
  if ( (unsigned int)v81 > 0x40 )
    sub_C43780((__int64)&v80, v68);
  else
    v80 = *v68;
  sub_C46A40((__int64)&v80, 1);
  v21 = (const void **)v69;
  v60 = (unsigned int)v81;
  v22 = *(_DWORD *)(v69 + 8);
  v74 = (int)v81;
  v58 = v80;
  v73 = v80;
  if ( v22 <= 0x40 )
  {
    v35 = *(const void **)v69;
    if ( v80 != *(const void **)v69 && v35 != *v68 )
      goto LABEL_38;
    v26 = *(_QWORD *)(v4 + 16);
    v25 = v68;
    if ( !v26 )
      goto LABEL_66;
  }
  else
  {
    v51 = v22;
    v55 = v69;
    v23 = sub_C43C50(v69, &v73);
    v21 = (const void **)v55;
    v22 = v51;
    if ( v23 )
    {
      v26 = *(_QWORD *)(v4 + 16);
      v25 = v68;
      if ( !v26 )
        goto LABEL_36;
    }
    else
    {
      v50 = v51;
      v52 = v68;
      v24 = sub_C43C50(v55, v68);
      v21 = (const void **)v55;
      v25 = v52;
      v22 = v50;
      if ( !v24 )
        goto LABEL_38;
      v26 = *(_QWORD *)(v4 + 16);
      if ( !v26 )
      {
LABEL_36:
        if ( sub_C43C50((__int64)v21, v25) )
          goto LABEL_37;
LABEL_38:
        v4 = 0;
        goto LABEL_39;
      }
    }
  }
  if ( !*(_QWORD *)(v26 + 8) )
  {
    v42 = *(_QWORD *)(v3 + 8);
    v57 = v42;
    v75 = sub_BD5D20(v3);
    v79 = 773;
    v76 = v43;
    v77 = ".biased";
    v53 = sub_AD8D80(v42, (__int64)v68);
    v44 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a2 + 80)
                                                                                                + 32LL))(
                     *(_QWORD *)(a2 + 80),
                     13,
                     v3,
                     v53,
                     0,
                     0);
    if ( !v44 )
    {
      v84 = 257;
      v44 = (_BYTE *)sub_B504D0(13, v3, v53, (__int64)&v80, 0, 0);
      (*(void (__fastcall **)(_QWORD, _BYTE *, const char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v44,
        &v75,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
      {
        v54 = v16;
        v46 = *(unsigned int **)a2;
        v47 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        do
        {
          v48 = *((_QWORD *)v46 + 1);
          v49 = *v46;
          v46 += 4;
          sub_B99FD0((__int64)v44, v49, v48);
        }
        while ( (unsigned int *)v47 != v46 );
        v16 = v54;
      }
    }
    v84 = 257;
    v45 = (_BYTE *)sub_AD8D80(v57, (__int64)v70);
    v4 = sub_A82350((unsigned int **)a2, v44, v45, (__int64)&v80);
    sub_BD6B90((unsigned __int8 *)v4, v8);
    goto LABEL_39;
  }
  v25 = v68;
  if ( v22 > 0x40 )
    goto LABEL_36;
  v35 = *v21;
LABEL_66:
  if ( *v25 != v35 )
    goto LABEL_38;
LABEL_37:
  if ( !(unsigned __int8)sub_98EF70(v4, v3) )
    goto LABEL_38;
LABEL_39:
  if ( v60 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  if ( v16 > 0x40 )
    goto LABEL_43;
  return v4;
}
