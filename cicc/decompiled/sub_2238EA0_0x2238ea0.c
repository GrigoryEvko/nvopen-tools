// Function: sub_2238EA0
// Address: 0x2238ea0
//
_QWORD *__fastcall sub_2238EA0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        int *a8)
{
  __int64 v9; // rbx
  int v10; // edx
  int v11; // r12d
  bool v12; // al
  int v13; // edx
  bool v14; // r15
  char v15; // al
  int v16; // edx
  char v17; // r13
  __int64 v18; // rsi
  int v19; // r14d
  char v20; // bp
  char *v21; // rax
  __int64 v22; // rdx
  int v23; // ebp
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // rsi
  char v27; // r15
  __int64 v28; // rcx
  int v29; // edx
  __int64 v30; // rdx
  _QWORD *v31; // rdi
  unsigned __int64 v32; // rax
  char v33; // al
  char v34; // r13
  bool v35; // r13
  __int64 v36; // rdx
  _QWORD *v37; // r12
  bool v39; // al
  char *v40; // rax
  int v41; // r8d
  char v42; // al
  void *v43; // rax
  int v44; // eax
  _QWORD *v45; // rdi
  unsigned __int64 v46; // rax
  int v47; // eax
  char v48; // al
  int v49; // eax
  int v50; // [rsp+4h] [rbp-A4h]
  int v51; // [rsp+4h] [rbp-A4h]
  unsigned int v52; // [rsp+4h] [rbp-A4h]
  unsigned __int8 v53; // [rsp+4h] [rbp-A4h]
  unsigned __int8 v54; // [rsp+4h] [rbp-A4h]
  char v55; // [rsp+8h] [rbp-A0h]
  int v56; // [rsp+10h] [rbp-98h]
  size_t n; // [rsp+28h] [rbp-80h]
  void *s; // [rsp+30h] [rbp-78h]
  unsigned int v59; // [rsp+38h] [rbp-70h]
  unsigned __int8 v60; // [rsp+38h] [rbp-70h]
  char v61; // [rsp+3Dh] [rbp-6Bh]
  unsigned __int8 v62; // [rsp+3Dh] [rbp-6Bh]
  unsigned __int8 v63; // [rsp+3Dh] [rbp-6Bh]
  char v64; // [rsp+3Eh] [rbp-6Ah]
  bool v65; // [rsp+3Fh] [rbp-69h]
  _QWORD *v66; // [rsp+40h] [rbp-68h] BYREF
  __int64 v67; // [rsp+48h] [rbp-60h]
  _QWORD *v68; // [rsp+50h] [rbp-58h] BYREF
  __int64 v69; // [rsp+58h] [rbp-50h]
  char v70; // [rsp+66h] [rbp-42h] BYREF
  __int64 v71[8]; // [rsp+68h] [rbp-40h] BYREF

  v68 = a2;
  v69 = a3;
  v66 = a4;
  v67 = a5;
  v9 = sub_2232A70((__int64)&v70, (__int64 *)(a6 + 208));
  v10 = *(_DWORD *)(a6 + 24) & 0x4A;
  if ( v10 == 64 )
  {
    v11 = 8;
LABEL_3:
    v50 = *(_DWORD *)(a6 + 24) & 0x4A;
    v12 = sub_2233E50((__int64)&v68, (__int64)&v66);
    v13 = v50;
    v14 = v12;
    if ( v12 )
    {
      v56 = v11;
      v18 = *(unsigned __int8 *)(v9 + 32);
      v17 = 0;
      n = v11;
      v19 = 0;
      v20 = 0;
      v65 = 0;
      goto LABEL_30;
    }
    goto LABEL_4;
  }
  v11 = 10;
  if ( v10 != 8 )
    goto LABEL_3;
  v11 = 16;
  v39 = sub_2233E50((__int64)&v68, (__int64)&v66);
  v13 = 8;
  v14 = v39;
  if ( v39 )
  {
    v65 = 0;
    v18 = *(unsigned __int8 *)(v9 + 32);
    v17 = 0;
    v19 = 0;
    n = 22;
    v20 = 0;
    v56 = 22;
    goto LABEL_30;
  }
LABEL_4:
  v51 = v13;
  v15 = sub_2233F00((__int64)&v68);
  v16 = v51;
  v17 = v15;
  v65 = *(_BYTE *)(v9 + 110) == (unsigned __int8)v15;
  if ( *(_BYTE *)(v9 + 110) == v15 || *(_BYTE *)(v9 + 111) == v15 )
  {
    v18 = *(unsigned __int8 *)(v9 + 32);
    if ( (!(_BYTE)v18 || *(_BYTE *)(v9 + 73) != v15) && *(_BYTE *)(v9 + 72) != v15 )
    {
      sub_22408B0(v68);
      LODWORD(v69) = -1;
      v14 = sub_2233E50((__int64)&v68, (__int64)&v66);
      if ( v14 )
      {
        v18 = *(unsigned __int8 *)(v9 + 32);
        v19 = 0;
        v20 = 0;
        if ( v11 != 16 )
          goto LABEL_29;
LABEL_10:
        n = 22;
        v56 = 22;
        goto LABEL_30;
      }
      v48 = sub_2233F00((__int64)&v68);
      v18 = *(unsigned __int8 *)(v9 + 32);
      v16 = v51;
      v17 = v48;
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
    v24 = v68;
    v19 = 0;
    v11 = 8;
    v25 = v68[2];
    if ( v25 < v68[3] )
    {
LABEL_23:
      LODWORD(v69) = -1;
      v24[2] = v25 + 1;
LABEL_24:
      v21 = (char *)v24[3];
      v26 = 0;
      if ( v24[2] >= (unsigned __int64)v21 )
      {
        v63 = v22;
        LODWORD(v21) = (*(__int64 (__fastcall **)(_QWORD *))(*v24 + 72LL))(v24);
        v26 = 0;
        v22 = v63;
        if ( (_DWORD)v21 == -1 )
        {
          v68 = 0;
          v26 = 1;
        }
      }
      goto LABEL_25;
    }
LABEL_59:
    v53 = v22;
    v21 = (char *)(*(__int64 (__fastcall **)(_QWORD *))(*v24 + 80LL))(v24);
    v24 = v68;
    LODWORD(v69) = -1;
    v22 = v53;
    if ( v68 )
      goto LABEL_24;
    v26 = 1;
LABEL_25:
    v27 = (_DWORD)v67 == -1;
    LOBYTE(v21) = v27 & (v66 != 0);
    v28 = (unsigned int)v21;
    if ( (_BYTE)v21 )
    {
      v21 = (char *)v66[3];
      v27 = 0;
      if ( v66[2] >= (unsigned __int64)v21 )
      {
        v60 = v22;
        v62 = v28;
        LODWORD(v21) = (*(__int64 (**)(void))(*v66 + 72LL))();
        v26 = (unsigned __int8)v26;
        v28 = v62;
        v22 = v60;
        if ( (_DWORD)v21 == -1 )
        {
          v66 = 0;
          v27 = v62;
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
    v17 = v69;
    if ( v68 && (_DWORD)v69 == -1 )
    {
      v21 = (char *)v68[2];
      if ( (unsigned __int64)v21 >= v68[3] )
      {
        v54 = v22;
        LODWORD(v21) = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*v68 + 72LL))(
                         v68,
                         v26,
                         v22,
                         v28);
        LODWORD(v22) = v54;
        v17 = (char)v21;
        if ( (_DWORD)v21 == -1 )
          v68 = 0;
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
      v56 = v11;
      v20 = 1;
      n = v11;
      goto LABEL_30;
    }
    v19 = 0;
    v22 = 0;
    v11 = 16;
LABEL_22:
    v24 = v68;
    v25 = v68[2];
    if ( v25 < v68[3] )
      goto LABEL_23;
    goto LABEL_59;
  }
  v14 = 0;
  v20 = 1;
LABEL_28:
  if ( v11 == 16 )
    goto LABEL_10;
LABEL_29:
  v56 = v11;
  n = v11;
LABEL_30:
  v71[0] = (__int64)&unk_4FD67D8;
  if ( (_BYTE)v18 )
  {
    v18 = 32;
    sub_2215AB0(v71, 0x20u);
  }
  v59 = 0xFFFFFFFF / v11;
  v64 = *(_BYTE *)(v9 + 136);
  if ( v64 )
  {
    if ( v14 )
    {
      v35 = v14;
      v52 = 0;
      v14 = 0;
      v61 = 0;
    }
    else
    {
      v61 = 0;
      s = (void *)(v9 + 114);
      v42 = *(_BYTE *)(v9 + 32);
      v52 = 0;
      if ( !v42 || *(_BYTE *)(v9 + 73) != v17 )
        goto LABEL_96;
LABEL_107:
      if ( v19 )
      {
        sub_2215DF0(v71, v19);
        v45 = v68;
        v19 = 0;
        v46 = v68[2];
        if ( v46 >= v68[3] )
        {
LABEL_109:
          (*(void (__fastcall **)(_QWORD *))(*v45 + 80LL))(v45);
          goto LABEL_104;
        }
        while ( 1 )
        {
          v45[2] = v46 + 1;
LABEL_104:
          LODWORD(v69) = -1;
          v35 = sub_2233E50((__int64)&v68, (__int64)&v66);
          if ( v35 )
            break;
          v17 = sub_2233F00((__int64)&v68);
          v42 = *(_BYTE *)(v9 + 32);
          if ( v42 && *(_BYTE *)(v9 + 73) == v17 )
            goto LABEL_107;
LABEL_96:
          if ( v17 == *(_BYTE *)(v9 + 72) )
            goto LABEL_92;
          v43 = memchr(s, v17, n);
          if ( !v43 )
            goto LABEL_92;
          v44 = (_DWORD)v43 - (_DWORD)s;
          if ( v44 > 15 )
            v44 -= 6;
          if ( v59 < v52 )
          {
            v61 = v64;
          }
          else
          {
            v61 |= ~v44 < v11 * v52;
            ++v19;
            v52 = v11 * v52 + v44;
          }
          v45 = v68;
          v46 = v68[2];
          if ( v46 >= v68[3] )
            goto LABEL_109;
        }
      }
      else
      {
        v35 = 0;
        v14 = v42;
      }
    }
  }
  else
  {
    v61 = 0;
    if ( v14 )
    {
      v35 = v14;
      v52 = 0;
      v14 = 0;
    }
    else
    {
      v52 = 0;
LABEL_35:
      v29 = v17;
      if ( v56 > 10 )
      {
        if ( (unsigned __int8)(v17 - 48) <= 9u )
          goto LABEL_38;
        while ( (unsigned __int8)(v17 - 97) <= 5u )
        {
          v30 = (unsigned int)(v29 - 87);
          if ( v59 < v52 )
            goto LABEL_40;
LABEL_68:
          v31 = v68;
          v61 |= ~(_DWORD)v30 < v11 * v52;
          ++v19;
          v52 = v30 + v11 * v52;
          v32 = v68[2];
          if ( v32 < v68[3] )
            goto LABEL_41;
LABEL_69:
          (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*v31 + 80LL))(v31, v18, v30);
          v31 = v68;
          LODWORD(v69) = -1;
          if ( v68 )
            goto LABEL_42;
          v55 = 1;
LABEL_43:
          v33 = (_DWORD)v67 == -1;
          v34 = v33 & (v66 != 0);
          if ( v34 )
          {
            v33 = 0;
            if ( v66[2] >= v66[3] )
            {
              v41 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v66 + 72LL))(v66, v18, v30);
              v33 = 0;
              if ( v41 == -1 )
              {
                v66 = 0;
                v33 = v34;
              }
            }
          }
          if ( v33 == v55 )
          {
            v35 = 1;
            goto LABEL_46;
          }
          v17 = v69;
          if ( (_DWORD)v69 != -1 || !v68 )
            goto LABEL_35;
          v40 = (char *)v68[2];
          if ( (unsigned __int64)v40 < v68[3] )
          {
            v17 = *v40;
            goto LABEL_35;
          }
          v29 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v68 + 72LL))(v68, v18, v30);
          v17 = v29;
          if ( v29 != -1 )
            goto LABEL_35;
          v68 = 0;
          if ( v56 <= 10 )
            goto LABEL_92;
          v17 = -1;
        }
        if ( (unsigned __int8)(v17 - 65) <= 5u )
        {
          v30 = (unsigned int)(v29 - 55);
          goto LABEL_39;
        }
      }
      else if ( v17 > 47 && v17 < (char)(n + 48) )
      {
LABEL_38:
        v30 = (unsigned int)(v17 - 48);
LABEL_39:
        if ( v59 >= v52 )
          goto LABEL_68;
LABEL_40:
        v31 = v68;
        v61 = 1;
        v32 = v68[2];
        if ( v32 >= v68[3] )
          goto LABEL_69;
LABEL_41:
        LODWORD(v69) = -1;
        v31[2] = v32 + 1;
LABEL_42:
        v55 = 0;
        if ( v31[2] >= v31[3] )
        {
          v55 = 0;
          if ( (*(unsigned int (__fastcall **)(_QWORD *, __int64, __int64))(*v31 + 72LL))(v31, v18, v30) == -1 )
          {
            v68 = 0;
            v55 = 1;
          }
        }
        goto LABEL_43;
      }
LABEL_92:
      v35 = 0;
    }
  }
LABEL_46:
  v36 = v71[0];
  if ( *(_QWORD *)(v71[0] - 24) )
  {
    sub_2215DF0(v71, v19);
    if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v9 + 16), *(_QWORD *)(v9 + 24), v71) )
      *a7 = 4;
    v36 = v71[0];
    if ( v19 || v20 == 1 || *(_QWORD *)(v71[0] - 24) )
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
    if ( v61 )
    {
      *a8 = -1;
      *a7 = 4;
    }
    else
    {
      v47 = -v52;
      if ( !v65 )
        v47 = v52;
      *a8 = v47;
    }
  }
  if ( v35 )
    *a7 |= 2u;
  v37 = v68;
  if ( (_UNKNOWN *)(v36 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v49 = _InterlockedExchangeAdd((volatile signed __int32 *)(v36 - 8), 0xFFFFFFFF);
    }
    else
    {
      v49 = *(_DWORD *)(v36 - 8);
      *(_DWORD *)(v36 - 8) = v49 - 1;
    }
    if ( v49 <= 0 )
      j_j___libc_free_0_1(v36 - 24);
  }
  return v37;
}
