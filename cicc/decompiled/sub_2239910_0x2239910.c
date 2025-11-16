// Function: sub_2239910
// Address: 0x2239910
//
_QWORD *__fastcall sub_2239910(
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
  bool v14; // r15
  char v15; // al
  int v16; // edx
  char v17; // bl
  __int64 v18; // rsi
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
  unsigned __int64 v29; // r13
  int v30; // edx
  __int64 v31; // rdx
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rax
  char v34; // al
  char v35; // bl
  bool v36; // bl
  __int64 v37; // rdx
  _QWORD *v38; // r12
  bool v40; // al
  bool v41; // cf
  char *v42; // rax
  int v43; // r8d
  char v44; // al
  void *v45; // rax
  int v46; // eax
  _QWORD *v47; // rdi
  unsigned __int64 v48; // rax
  char v49; // al
  int v50; // eax
  unsigned __int64 v51; // [rsp+8h] [rbp-B0h]
  unsigned __int8 v52; // [rsp+8h] [rbp-B0h]
  unsigned __int8 v53; // [rsp+8h] [rbp-B0h]
  char v54; // [rsp+10h] [rbp-A8h]
  int v55; // [rsp+18h] [rbp-A0h]
  unsigned __int8 v56; // [rsp+18h] [rbp-A0h]
  __int64 v57; // [rsp+28h] [rbp-90h]
  size_t n; // [rsp+38h] [rbp-80h]
  void *s; // [rsp+40h] [rbp-78h]
  int v60; // [rsp+48h] [rbp-70h]
  int v61; // [rsp+48h] [rbp-70h]
  char v62; // [rsp+48h] [rbp-70h]
  unsigned __int8 v63; // [rsp+48h] [rbp-70h]
  unsigned __int8 v64; // [rsp+48h] [rbp-70h]
  char v65; // [rsp+4Eh] [rbp-6Ah]
  bool v66; // [rsp+4Fh] [rbp-69h]
  _QWORD *v67; // [rsp+50h] [rbp-68h] BYREF
  __int64 v68; // [rsp+58h] [rbp-60h]
  _QWORD *v69; // [rsp+60h] [rbp-58h] BYREF
  __int64 v70; // [rsp+68h] [rbp-50h]
  char v71; // [rsp+76h] [rbp-42h] BYREF
  __int64 v72[8]; // [rsp+78h] [rbp-40h] BYREF

  v69 = a2;
  v70 = a3;
  v67 = a4;
  v68 = a5;
  v9 = sub_2232A70((__int64)&v71, (__int64 *)(a6 + 208));
  v10 = *(_DWORD *)(a6 + 24) & 0x4A;
  if ( v10 == 64 )
  {
    v11 = 8;
LABEL_3:
    v60 = *(_DWORD *)(a6 + 24) & 0x4A;
    v12 = sub_2233E50((__int64)&v69, (__int64)&v67);
    v13 = v60;
    v14 = v12;
    if ( v12 )
    {
      v55 = v11;
      v18 = *(unsigned __int8 *)(v9 + 32);
      v17 = 0;
      n = v11;
      v19 = 0;
      v20 = 0;
      v66 = 0;
      goto LABEL_30;
    }
    goto LABEL_4;
  }
  v11 = 10;
  if ( v10 != 8 )
    goto LABEL_3;
  v11 = 16;
  v40 = sub_2233E50((__int64)&v69, (__int64)&v67);
  v13 = 8;
  v14 = v40;
  if ( v40 )
  {
    v66 = 0;
    v18 = *(unsigned __int8 *)(v9 + 32);
    v17 = 0;
    v19 = 0;
    n = 22;
    v20 = 0;
    v55 = 22;
    goto LABEL_30;
  }
LABEL_4:
  v61 = v13;
  v15 = sub_2233F00((__int64)&v69);
  v16 = v61;
  v17 = v15;
  v66 = *(_BYTE *)(v9 + 110) == (unsigned __int8)v15;
  if ( *(_BYTE *)(v9 + 110) == v15 || *(_BYTE *)(v9 + 111) == v15 )
  {
    v18 = *(unsigned __int8 *)(v9 + 32);
    if ( (!(_BYTE)v18 || *(_BYTE *)(v9 + 73) != v15) && *(_BYTE *)(v9 + 72) != v15 )
    {
      sub_22408B0(v69);
      LODWORD(v70) = -1;
      v14 = sub_2233E50((__int64)&v69, (__int64)&v67);
      if ( v14 )
      {
        v18 = *(unsigned __int8 *)(v9 + 32);
        v19 = 0;
        v20 = 0;
        if ( v11 != 16 )
          goto LABEL_29;
LABEL_10:
        n = 22;
        v55 = 22;
        goto LABEL_30;
      }
      v49 = sub_2233F00((__int64)&v69);
      v18 = *(unsigned __int8 *)(v9 + 32);
      v16 = v61;
      v17 = v49;
    }
  }
  else
  {
    v18 = *(unsigned __int8 *)(v9 + 32);
  }
  LODWORD(v21) = v16;
  v19 = 0;
  LODWORD(v22) = 0;
  v23 = (int)v21;
  while ( 1 )
  {
    if ( (_BYTE)v18 && *(_BYTE *)(v9 + 73) == v17 || *(_BYTE *)(v9 + 72) == v17 )
    {
LABEL_64:
      v20 = v22;
      v14 = 0;
      goto LABEL_28;
    }
    if ( *(_BYTE *)(v9 + 114) != v17 )
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
    v24 = v69;
    v19 = 0;
    v11 = 8;
    v25 = v69[2];
    if ( v25 < v69[3] )
    {
LABEL_23:
      LODWORD(v70) = -1;
      v24[2] = v25 + 1;
LABEL_24:
      v21 = (char *)v24[3];
      v26 = 0;
      if ( v24[2] >= (unsigned __int64)v21 )
      {
        v53 = v22;
        LODWORD(v21) = (*(__int64 (__fastcall **)(_QWORD *))(*v24 + 72LL))(v24);
        v26 = 0;
        v22 = v53;
        if ( (_DWORD)v21 == -1 )
        {
          v69 = 0;
          v26 = 1;
        }
      }
      goto LABEL_25;
    }
LABEL_59:
    v63 = v22;
    v21 = (char *)(*(__int64 (__fastcall **)(_QWORD *))(*v24 + 80LL))(v24);
    v24 = v69;
    LODWORD(v70) = -1;
    v22 = v63;
    if ( v69 )
      goto LABEL_24;
    v26 = 1;
LABEL_25:
    v27 = (_DWORD)v68 == -1;
    LOBYTE(v21) = v27 & (v67 != 0);
    v28 = (unsigned int)v21;
    if ( (_BYTE)v21 )
    {
      v21 = (char *)v67[3];
      v27 = 0;
      if ( v67[2] >= (unsigned __int64)v21 )
      {
        v56 = v22;
        v52 = v28;
        LODWORD(v21) = (*(__int64 (**)(void))(*v67 + 72LL))();
        v26 = (unsigned __int8)v26;
        v28 = v52;
        v22 = v56;
        if ( (_DWORD)v21 == -1 )
        {
          v67 = 0;
          v27 = v52;
        }
      }
    }
    if ( v27 == (_BYTE)v26 )
    {
      v18 = *(unsigned __int8 *)(v9 + 32);
      v20 = v22;
      v14 = 1;
      goto LABEL_28;
    }
    v17 = v70;
    if ( v69 && (_DWORD)v70 == -1 )
    {
      v21 = (char *)v69[2];
      if ( (unsigned __int64)v21 >= v69[3] )
      {
        v64 = v22;
        LODWORD(v21) = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*v69 + 72LL))(
                         v69,
                         v26,
                         v22,
                         v28);
        LODWORD(v22) = v64;
        v17 = (char)v21;
        if ( (_DWORD)v21 == -1 )
          v69 = 0;
      }
      else
      {
        v17 = *v21;
      }
    }
    v18 = *(unsigned __int8 *)(v9 + 32);
    if ( !(_BYTE)v22 )
      goto LABEL_64;
  }
  if ( !(_BYTE)v22 )
    goto LABEL_64;
LABEL_18:
  if ( *(_BYTE *)(v9 + 112) == v17 || *(_BYTE *)(v9 + 113) == v17 )
  {
    if ( v23 != 0 && v11 != 16 )
    {
      v14 = 0;
      v55 = v11;
      v20 = 1;
      n = v11;
      goto LABEL_30;
    }
    v19 = 0;
    v22 = 0;
    v11 = 16;
LABEL_22:
    v24 = v69;
    v25 = v69[2];
    if ( v25 < v69[3] )
      goto LABEL_23;
    goto LABEL_59;
  }
  v14 = 0;
  v20 = 1;
LABEL_28:
  if ( v11 == 16 )
    goto LABEL_10;
LABEL_29:
  v55 = v11;
  n = v11;
LABEL_30:
  v72[0] = (__int64)&unk_4FD67D8;
  if ( (_BYTE)v18 )
  {
    v18 = 32;
    sub_2215AB0(v72, 0x20u);
  }
  v57 = v11;
  v51 = 0xFFFFFFFFFFFFFFFFLL / v11;
  v65 = *(_BYTE *)(v9 + 136);
  if ( v65 )
  {
    if ( v14 )
    {
      v36 = v14;
      v62 = 0;
      v29 = 0;
      v14 = 0;
    }
    else
    {
      v62 = 0;
      v29 = 0;
      s = (void *)(v9 + 114);
      v44 = *(_BYTE *)(v9 + 32);
      if ( !v44 || *(_BYTE *)(v9 + 73) != v17 )
        goto LABEL_96;
LABEL_107:
      if ( v19 )
      {
        sub_2215DF0(v72, v19);
        v47 = v69;
        v19 = 0;
        v48 = v69[2];
        if ( v48 >= v69[3] )
        {
LABEL_109:
          (*(void (__fastcall **)(_QWORD *))(*v47 + 80LL))(v47);
          goto LABEL_104;
        }
        while ( 1 )
        {
          v47[2] = v48 + 1;
LABEL_104:
          LODWORD(v70) = -1;
          v36 = sub_2233E50((__int64)&v69, (__int64)&v67);
          if ( v36 )
            break;
          v17 = sub_2233F00((__int64)&v69);
          v44 = *(_BYTE *)(v9 + 32);
          if ( v44 && *(_BYTE *)(v9 + 73) == v17 )
            goto LABEL_107;
LABEL_96:
          if ( v17 == *(_BYTE *)(v9 + 72) )
            goto LABEL_92;
          v45 = memchr(s, v17, n);
          if ( !v45 )
            goto LABEL_92;
          v46 = (_DWORD)v45 - (_DWORD)s;
          if ( v46 > 15 )
            v46 -= 6;
          if ( v51 < v29 )
          {
            v62 = v65;
          }
          else
          {
            v41 = ~(__int64)v46 < v29 * v57;
            v29 = v46 + v29 * v57;
            v62 |= v41;
            ++v19;
          }
          v47 = v69;
          v48 = v69[2];
          if ( v48 >= v69[3] )
            goto LABEL_109;
        }
      }
      else
      {
        v36 = 0;
        v14 = v44;
      }
    }
  }
  else
  {
    v62 = 0;
    if ( v14 )
    {
      v36 = v14;
      v29 = 0;
      v14 = 0;
    }
    else
    {
      v29 = 0;
LABEL_35:
      v30 = v17;
      if ( v55 > 10 )
      {
        if ( (unsigned __int8)(v17 - 48) <= 9u )
          goto LABEL_38;
        while ( (unsigned __int8)(v17 - 97) <= 5u )
        {
          v31 = (unsigned int)(v30 - 87);
          if ( v51 < v29 )
            goto LABEL_40;
LABEL_68:
          v31 = (int)v31;
          v32 = v69;
          v41 = ~(__int64)(int)v31 < v29 * v57;
          v29 = (int)v31 + v29 * v57;
          v33 = v69[2];
          ++v19;
          v62 |= v41;
          if ( v33 < v69[3] )
            goto LABEL_41;
LABEL_69:
          (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*v32 + 80LL))(v32, v18, v31);
          v32 = v69;
          LODWORD(v70) = -1;
          if ( v69 )
            goto LABEL_42;
          v54 = 1;
LABEL_43:
          v34 = (_DWORD)v68 == -1;
          v35 = v34 & (v67 != 0);
          if ( v35 )
          {
            v34 = 0;
            if ( v67[2] >= v67[3] )
            {
              v43 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v67 + 72LL))(v67, v18, v31);
              v34 = 0;
              if ( v43 == -1 )
              {
                v67 = 0;
                v34 = v35;
              }
            }
          }
          if ( v34 == v54 )
          {
            v36 = 1;
            goto LABEL_46;
          }
          v17 = v70;
          if ( (_DWORD)v70 != -1 || !v69 )
            goto LABEL_35;
          v42 = (char *)v69[2];
          if ( (unsigned __int64)v42 < v69[3] )
          {
            v17 = *v42;
            goto LABEL_35;
          }
          v30 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v69 + 72LL))(v69, v18, v31);
          v17 = v30;
          if ( v30 != -1 )
            goto LABEL_35;
          v69 = 0;
          if ( v55 <= 10 )
            goto LABEL_92;
          v17 = -1;
        }
        if ( (unsigned __int8)(v17 - 65) <= 5u )
        {
          v31 = (unsigned int)(v30 - 55);
          goto LABEL_39;
        }
      }
      else if ( v17 > 47 && v17 < (char)(n + 48) )
      {
LABEL_38:
        v31 = (unsigned int)(v17 - 48);
LABEL_39:
        if ( v51 >= v29 )
          goto LABEL_68;
LABEL_40:
        v32 = v69;
        v62 = 1;
        v33 = v69[2];
        if ( v33 >= v69[3] )
          goto LABEL_69;
LABEL_41:
        LODWORD(v70) = -1;
        v32[2] = v33 + 1;
LABEL_42:
        v54 = 0;
        if ( v32[2] >= v32[3] )
        {
          v54 = 0;
          if ( (*(unsigned int (__fastcall **)(_QWORD *, __int64, __int64))(*v32 + 72LL))(v32, v18, v31) == -1 )
          {
            v69 = 0;
            v54 = 1;
          }
        }
        goto LABEL_43;
      }
LABEL_92:
      v36 = 0;
    }
  }
LABEL_46:
  v37 = v72[0];
  if ( *(_QWORD *)(v72[0] - 24) )
  {
    sub_2215DF0(v72, v19);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v72) )
      *a7 = 4;
    v37 = v72[0];
    if ( v19 || v20 == 1 || *(_QWORD *)(v72[0] - 24) )
      goto LABEL_83;
LABEL_49:
    *a8 = 0;
    *a7 = 4;
  }
  else
  {
    if ( v20 != 1 && !v19 )
      goto LABEL_49;
LABEL_83:
    if ( v14 )
      goto LABEL_49;
    if ( v62 )
    {
      *a8 = -1;
      *a7 = 4;
    }
    else
    {
      if ( v66 )
        v29 = -(__int64)v29;
      *a8 = v29;
    }
  }
  if ( v36 )
    *a7 |= 2u;
  v38 = v69;
  if ( (_UNKNOWN *)(v37 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v50 = _InterlockedExchangeAdd((volatile signed __int32 *)(v37 - 8), 0xFFFFFFFF);
    }
    else
    {
      v50 = *(_DWORD *)(v37 - 8);
      *(_DWORD *)(v37 - 8) = v50 - 1;
    }
    if ( v50 <= 0 )
      j_j___libc_free_0_1(v37 - 24);
  }
  return v38;
}
