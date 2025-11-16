// Function: sub_223A410
// Address: 0x223a410
//
_QWORD *__fastcall sub_223A410(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        unsigned __int64 *a8)
{
  __int64 v9; // rbp
  int v10; // edx
  int v11; // r13d
  bool v12; // al
  int v13; // edx
  char v14; // al
  int v15; // edx
  char v16; // bl
  __int64 v17; // rsi
  char v18; // r15
  int v19; // r14d
  char v20; // r12
  char *v21; // rax
  __int64 v22; // rdx
  int v23; // r12d
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // rsi
  char v27; // r15
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  int v30; // edx
  __int64 v31; // rdx
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rax
  char v34; // r13
  char v35; // al
  char v36; // bl
  bool v37; // bl
  __int64 v38; // rdx
  _QWORD *v39; // r12
  bool v41; // al
  unsigned __int64 v42; // rax
  char *v43; // rax
  unsigned __int64 v44; // rax
  int v45; // r8d
  char v46; // al
  void *v47; // rax
  int v48; // eax
  unsigned __int64 v49; // rsi
  _QWORD *v50; // rdi
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  char v53; // al
  int v54; // eax
  int v55; // [rsp+0h] [rbp-C8h]
  int v56; // [rsp+0h] [rbp-C8h]
  unsigned __int64 v57; // [rsp+0h] [rbp-C8h]
  unsigned __int8 v58; // [rsp+0h] [rbp-C8h]
  unsigned __int8 v59; // [rsp+0h] [rbp-C8h]
  int v60; // [rsp+8h] [rbp-C0h]
  unsigned __int8 v61; // [rsp+8h] [rbp-C0h]
  unsigned __int64 v62; // [rsp+10h] [rbp-B8h]
  unsigned __int64 v63; // [rsp+28h] [rbp-A0h]
  __int64 v64; // [rsp+30h] [rbp-98h]
  size_t n; // [rsp+40h] [rbp-88h]
  unsigned __int64 v66; // [rsp+48h] [rbp-80h]
  char v67; // [rsp+56h] [rbp-72h]
  unsigned __int8 v68; // [rsp+56h] [rbp-72h]
  unsigned __int8 v69; // [rsp+56h] [rbp-72h]
  bool v70; // [rsp+57h] [rbp-71h]
  _QWORD *v71; // [rsp+60h] [rbp-68h] BYREF
  __int64 v72; // [rsp+68h] [rbp-60h]
  _QWORD *v73; // [rsp+70h] [rbp-58h] BYREF
  __int64 v74; // [rsp+78h] [rbp-50h]
  char v75; // [rsp+86h] [rbp-42h] BYREF
  __int64 v76[8]; // [rsp+88h] [rbp-40h] BYREF

  v73 = a2;
  v74 = a3;
  v71 = a4;
  v72 = a5;
  v9 = sub_2232A70((__int64)&v75, (__int64 *)(a6 + 208));
  v10 = *(_DWORD *)(a6 + 24) & 0x4A;
  if ( v10 == 64 )
  {
    v11 = 8;
    goto LABEL_3;
  }
  v11 = 10;
  if ( v10 != 8 )
  {
LABEL_3:
    v55 = *(_DWORD *)(a6 + 24) & 0x4A;
    v12 = sub_2233E50((__int64)&v73, (__int64)&v71);
    v13 = v55;
    if ( !v12 )
      goto LABEL_4;
    v20 = *(_BYTE *)(v9 + 32);
    n = v11;
    v42 = v11;
LABEL_97:
    v76[0] = (__int64)&unk_4FD67D8;
    if ( !v20 )
    {
      v63 = v42;
      v18 = *(_BYTE *)(v9 + 136);
      v70 = 0;
      v66 = 0x7FFFFFFFFFFFFFFFLL / v42;
      if ( !v18 )
      {
        v67 = 0;
        v19 = 0;
        v57 = 0;
LABEL_46:
        v37 = 1;
        goto LABEL_47;
      }
      v64 = 0x7FFFFFFFFFFFFFFFLL;
      v19 = 0;
      v16 = 0;
      goto LABEL_100;
    }
    v70 = 0;
    v18 = v20;
    v16 = 0;
    v19 = 0;
    v60 = v11;
    v20 = 0;
    v11 = v42;
    goto LABEL_122;
  }
  v11 = 16;
  v41 = sub_2233E50((__int64)&v73, (__int64)&v71);
  v13 = 8;
  if ( v41 )
  {
    v20 = *(_BYTE *)(v9 + 32);
    v42 = 16;
    n = 22;
    v11 = 22;
    goto LABEL_97;
  }
LABEL_4:
  v56 = v13;
  v14 = sub_2233F00((__int64)&v73);
  v15 = v56;
  v16 = v14;
  v70 = *(_BYTE *)(v9 + 110) == (unsigned __int8)v14;
  if ( *(_BYTE *)(v9 + 110) == v14 || *(_BYTE *)(v9 + 111) == v14 )
  {
    v17 = *(unsigned __int8 *)(v9 + 32);
    if ( (!(_BYTE)v17 || *(_BYTE *)(v9 + 73) != v14) && *(_BYTE *)(v9 + 72) != v14 )
    {
      sub_22408B0(v73);
      LODWORD(v74) = -1;
      v18 = sub_2233E50((__int64)&v73, (__int64)&v71);
      if ( v18 )
      {
        v17 = *(unsigned __int8 *)(v9 + 32);
        v19 = 0;
        v20 = 0;
        if ( v11 != 16 )
          goto LABEL_29;
LABEL_10:
        n = 22;
        v60 = 22;
        goto LABEL_30;
      }
      v53 = sub_2233F00((__int64)&v73);
      v17 = *(unsigned __int8 *)(v9 + 32);
      v15 = v56;
      v16 = v53;
    }
  }
  else
  {
    v17 = *(unsigned __int8 *)(v9 + 32);
  }
  LODWORD(v21) = v15;
  v19 = 0;
  LODWORD(v22) = 0;
  v23 = (int)v21;
  while ( 1 )
  {
    if ( (_BYTE)v17 && *(_BYTE *)(v9 + 73) == v16 || *(_BYTE *)(v9 + 72) == v16 )
    {
LABEL_65:
      v20 = v22;
      v18 = 0;
      goto LABEL_28;
    }
    if ( *(_BYTE *)(v9 + 114) != v16 )
      break;
    v22 = (unsigned int)v22 ^ 1;
    LOBYTE(v21) = v22 | (v11 == 10);
    if ( !(_BYTE)v21 )
      goto LABEL_18;
    LOBYTE(v22) = v23 == 0 || v11 == 8;
    if ( !(_BYTE)v22 )
    {
      ++v19;
      v22 = (unsigned int)v21;
      goto LABEL_22;
    }
    v24 = v73;
    v19 = 0;
    v11 = 8;
    v25 = v73[2];
    if ( v25 < v73[3] )
    {
LABEL_23:
      LODWORD(v74) = -1;
      v24[2] = v25 + 1;
LABEL_24:
      v21 = (char *)v24[3];
      v26 = 0;
      if ( v24[2] >= (unsigned __int64)v21 )
      {
        v69 = v22;
        LODWORD(v21) = (*(__int64 (__fastcall **)(_QWORD *))(*v24 + 72LL))(v24);
        v26 = 0;
        v22 = v69;
        if ( (_DWORD)v21 == -1 )
        {
          v73 = 0;
          v26 = 1;
        }
      }
      goto LABEL_25;
    }
LABEL_60:
    v58 = v22;
    v21 = (char *)(*(__int64 (__fastcall **)(_QWORD *))(*v24 + 80LL))(v24);
    v24 = v73;
    LODWORD(v74) = -1;
    v22 = v58;
    if ( v73 )
      goto LABEL_24;
    v26 = 1;
LABEL_25:
    v27 = (_DWORD)v72 == -1;
    LOBYTE(v21) = v27 & (v71 != 0);
    v28 = (unsigned int)v21;
    if ( (_BYTE)v21 )
    {
      v21 = (char *)v71[3];
      v27 = 0;
      if ( v71[2] >= (unsigned __int64)v21 )
      {
        v61 = v22;
        v68 = v28;
        LODWORD(v21) = (*(__int64 (**)(void))(*v71 + 72LL))();
        v26 = (unsigned __int8)v26;
        v28 = v68;
        v22 = v61;
        if ( (_DWORD)v21 == -1 )
        {
          v71 = 0;
          v27 = v68;
        }
      }
    }
    if ( (_BYTE)v26 == v27 )
    {
      v17 = *(unsigned __int8 *)(v9 + 32);
      v20 = v22;
      v18 = 1;
      goto LABEL_28;
    }
    v16 = v74;
    if ( v73 && (_DWORD)v74 == -1 )
    {
      v21 = (char *)v73[2];
      if ( (unsigned __int64)v21 >= v73[3] )
      {
        v59 = v22;
        LODWORD(v21) = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*v73 + 72LL))(
                         v73,
                         v26,
                         v22,
                         v28);
        LODWORD(v22) = v59;
        v16 = (char)v21;
        if ( (_DWORD)v21 == -1 )
          v73 = 0;
      }
      else
      {
        v16 = *v21;
      }
    }
    v17 = *(unsigned __int8 *)(v9 + 32);
    if ( !(_BYTE)v22 )
      goto LABEL_65;
  }
  if ( !(_BYTE)v22 )
    goto LABEL_65;
LABEL_18:
  if ( *(_BYTE *)(v9 + 112) == v16 || *(_BYTE *)(v9 + 113) == v16 )
  {
    if ( v23 != 0 && v11 != 16 )
    {
      v18 = 0;
      v60 = v11;
      v20 = 1;
      n = v11;
      goto LABEL_30;
    }
    v19 = 0;
    v22 = 0;
    v11 = 16;
LABEL_22:
    v24 = v73;
    v25 = v73[2];
    if ( v25 < v73[3] )
      goto LABEL_23;
    goto LABEL_60;
  }
  v18 = 0;
  v20 = 1;
LABEL_28:
  if ( v11 == 16 )
    goto LABEL_10;
LABEL_29:
  v60 = v11;
  n = v11;
LABEL_30:
  v76[0] = (__int64)&unk_4FD67D8;
  if ( (_BYTE)v17 )
  {
LABEL_122:
    v17 = 32;
    sub_2215AB0(v76, 0x20u);
  }
  v29 = 0x7FFFFFFFFFFFFFFFLL;
  if ( v70 )
    v29 = 0x8000000000000000LL;
  v63 = v11;
  v64 = v29;
  v62 = v29 / v11;
  v66 = v62;
  v67 = *(_BYTE *)(v9 + 136);
  if ( !v67 )
  {
    if ( v18 )
    {
      v37 = v18;
      v57 = 0;
      v18 = 0;
    }
    else
    {
      v57 = 0;
LABEL_36:
      v30 = v16;
      if ( v60 > 10 )
      {
        if ( (unsigned __int8)(v16 - 48) <= 9u )
          goto LABEL_39;
        while ( (unsigned __int8)(v16 - 97) <= 5u )
        {
          v31 = (unsigned int)(v30 - 87);
          if ( v62 < v57 )
            goto LABEL_41;
LABEL_69:
          v31 = (int)v31;
          v32 = v73;
          v67 |= v64 - (int)v31 < v63 * v57;
          ++v19;
          v57 = (int)v31 + v63 * v57;
          v33 = v73[2];
          if ( v33 < v73[3] )
            goto LABEL_42;
LABEL_70:
          (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*v32 + 80LL))(v32, v17, v31);
          v32 = v73;
          LODWORD(v74) = -1;
          if ( v73 )
            goto LABEL_43;
          v34 = 1;
LABEL_44:
          v35 = (_DWORD)v72 == -1;
          v36 = v35 & (v71 != 0);
          if ( v36 )
          {
            v35 = 0;
            if ( v71[2] >= v71[3] )
            {
              v45 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v71 + 72LL))(v71, v17, v31);
              v35 = 0;
              if ( v45 == -1 )
              {
                v71 = 0;
                v35 = v36;
              }
            }
          }
          if ( v34 == v35 )
            goto LABEL_46;
          v16 = v74;
          if ( (_DWORD)v74 != -1 || !v73 )
            goto LABEL_36;
          v43 = (char *)v73[2];
          if ( (unsigned __int64)v43 < v73[3] )
          {
            v16 = *v43;
            goto LABEL_36;
          }
          v30 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v73 + 72LL))(v73, v17, v31);
          v16 = v30;
          if ( v30 != -1 )
            goto LABEL_36;
          v73 = 0;
          if ( v60 <= 10 )
            goto LABEL_95;
          v16 = -1;
        }
        if ( (unsigned __int8)(v16 - 65) <= 5u )
        {
          v31 = (unsigned int)(v30 - 55);
          goto LABEL_40;
        }
      }
      else if ( v16 > 47 && v16 < (char)(n + 48) )
      {
LABEL_39:
        v31 = (unsigned int)(v16 - 48);
LABEL_40:
        if ( v62 >= v57 )
          goto LABEL_69;
LABEL_41:
        v32 = v73;
        v67 = 1;
        v33 = v73[2];
        if ( v33 >= v73[3] )
          goto LABEL_70;
LABEL_42:
        LODWORD(v74) = -1;
        v32[2] = v33 + 1;
LABEL_43:
        v34 = 0;
        if ( v32[2] >= v32[3]
          && (*(unsigned int (__fastcall **)(_QWORD *, __int64, __int64))(*v32 + 72LL))(v32, v17, v31) == -1 )
        {
          v73 = 0;
          v34 = 1;
        }
        goto LABEL_44;
      }
LABEL_95:
      v37 = 0;
    }
    goto LABEL_47;
  }
LABEL_100:
  if ( v18 )
  {
    v37 = v18;
    v57 = 0;
    v18 = 0;
    v67 = 0;
  }
  else
  {
    v67 = 0;
    v46 = *(_BYTE *)(v9 + 32);
    v57 = 0;
    if ( !v46 || *(_BYTE *)(v9 + 73) != v16 )
      goto LABEL_103;
LABEL_114:
    if ( v19 )
    {
      v49 = (unsigned int)(char)v19;
      sub_2215DF0(v76, v19);
      v50 = v73;
      v19 = 0;
      v51 = v73[2];
      if ( v51 >= v73[3] )
      {
LABEL_116:
        (*(void (__fastcall **)(_QWORD *, unsigned __int64))(*v50 + 80LL))(v50, v49);
        goto LABEL_111;
      }
      while ( 1 )
      {
        v50[2] = v51 + 1;
LABEL_111:
        LODWORD(v74) = -1;
        v37 = sub_2233E50((__int64)&v73, (__int64)&v71);
        if ( v37 )
          break;
        v16 = sub_2233F00((__int64)&v73);
        v46 = *(_BYTE *)(v9 + 32);
        if ( v46 && *(_BYTE *)(v9 + 73) == v16 )
          goto LABEL_114;
LABEL_103:
        if ( v16 == *(_BYTE *)(v9 + 72) )
          goto LABEL_95;
        v47 = memchr((const void *)(v9 + 114), v16, n);
        if ( !v47 )
          goto LABEL_95;
        v48 = (_DWORD)v47 - (v9 + 114);
        v49 = v66;
        if ( v48 > 15 )
          v48 -= 6;
        if ( v57 > v66 )
        {
          v67 = 1;
        }
        else
        {
          v67 |= v64 - v48 < v57 * v63;
          ++v19;
          v57 = v57 * v63 + v48;
        }
        v50 = v73;
        v51 = v73[2];
        if ( v51 >= v73[3] )
          goto LABEL_116;
      }
    }
    else
    {
      v37 = 0;
      v18 = v46;
    }
  }
LABEL_47:
  v38 = v76[0];
  if ( *(_QWORD *)(v76[0] - 24) )
  {
    sub_2215DF0(v76, v19);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v76) )
      *a7 = 4;
    v38 = v76[0];
    if ( v19 || v20 == 1 || *(_QWORD *)(v76[0] - 24) )
      goto LABEL_84;
LABEL_50:
    *a8 = 0;
    *a7 = 4;
  }
  else
  {
    if ( v20 != 1 && !v19 )
      goto LABEL_50;
LABEL_84:
    if ( v18 )
      goto LABEL_50;
    if ( v67 )
    {
      v44 = 0x8000000000000000LL;
      if ( !v70 )
        v44 = 0x7FFFFFFFFFFFFFFFLL;
      *a8 = v44;
      *a7 = 4;
    }
    else
    {
      v52 = -(__int64)v57;
      if ( !v70 )
        v52 = v57;
      *a8 = v52;
    }
  }
  if ( v37 )
    *a7 |= 2u;
  v39 = v73;
  if ( (_UNKNOWN *)(v38 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v54 = _InterlockedExchangeAdd((volatile signed __int32 *)(v38 - 8), 0xFFFFFFFF);
    }
    else
    {
      v54 = *(_DWORD *)(v38 - 8);
      *(_DWORD *)(v38 - 8) = v54 - 1;
    }
    if ( v54 <= 0 )
      j_j___libc_free_0_1(v38 - 24);
  }
  return v39;
}
