// Function: sub_224F550
// Address: 0x224f550
//
_QWORD *__fastcall sub_224F550(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        __int64 *a8)
{
  _QWORD *v8; // rbp
  __int64 v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // r14
  bool v13; // zf
  int v14; // r15d
  __int64 v15; // rbp
  wchar_t v16; // eax
  _QWORD *v17; // rdi
  unsigned __int64 v18; // r13
  char v19; // al
  char v20; // bp
  wchar_t v21; // eax
  unsigned __int64 v22; // rax
  char v23; // bl
  char v24; // bp
  _DWORD *v25; // rax
  int v26; // eax
  wchar_t v27; // eax
  _QWORD *v28; // rdi
  char v29; // r13
  __int64 v30; // rbp
  volatile signed __int32 *v31; // rax
  unsigned __int64 v32; // rax
  char v33; // bl
  char v34; // bp
  char v35; // al
  char v36; // bp
  wchar_t v37; // ebx
  wchar_t *v38; // rax
  __int64 v39; // rbp
  volatile signed __int32 *v40; // rax
  char v41; // bl
  unsigned __int64 v42; // rbp
  unsigned __int64 v43; // rax
  wchar_t v44; // eax
  wchar_t *v45; // rax
  wchar_t v46; // eax
  _QWORD *v47; // rdi
  char v48; // al
  char v49; // r13
  unsigned __int64 v50; // rax
  char v51; // bl
  char v52; // r13
  _DWORD *v53; // rax
  int v54; // eax
  int *v55; // rax
  int v56; // edx
  bool v57; // al
  _QWORD *v58; // r12
  volatile signed __int32 *v59; // rdi
  unsigned __int64 v60; // rdi
  _DWORD *v62; // rax
  int v63; // eax
  wchar_t *v64; // rax
  wchar_t v65; // eax
  _DWORD *v66; // rax
  int v67; // edx
  _DWORD *v68; // rax
  int v69; // eax
  unsigned int v70; // eax
  int *v71; // rax
  int v72; // edx
  char v73; // si
  _DWORD *v74; // rax
  unsigned __int64 v75; // rax
  unsigned __int64 v76; // rdx
  __int64 v77; // r13
  volatile signed __int32 *v78; // rax
  int v79; // eax
  int v80; // eax
  wchar_t *s; // [rsp+0h] [rbp-D8h]
  __int64 v82; // [rsp+8h] [rbp-D0h]
  unsigned __int64 v83; // [rsp+18h] [rbp-C0h]
  __int64 v84; // [rsp+30h] [rbp-A8h]
  unsigned __int64 v85; // [rsp+48h] [rbp-90h]
  char v87; // [rsp+58h] [rbp-80h]
  char v88; // [rsp+5Dh] [rbp-7Bh]
  char v89; // [rsp+5Eh] [rbp-7Ah]
  bool v90; // [rsp+5Fh] [rbp-79h]
  _QWORD *v91; // [rsp+60h] [rbp-78h] BYREF
  __int64 v92; // [rsp+68h] [rbp-70h]
  _QWORD *v93; // [rsp+70h] [rbp-68h] BYREF
  wchar_t c[4]; // [rsp+78h] [rbp-60h]
  int v95; // [rsp+8Ch] [rbp-4Ch]
  volatile signed __int32 *v96; // [rsp+90h] [rbp-48h] BYREF
  volatile signed __int32 *v97[8]; // [rsp+98h] [rbp-40h] BYREF

  v8 = (_QWORD *)(a6 + 208);
  v93 = a2;
  *(_QWORD *)c = a3;
  v91 = a4;
  v92 = a5;
  v84 = sub_2243120((_QWORD *)(a6 + 208), (__int64)a2);
  v10 = sub_22091A0(&qword_4FD6A90);
  v11 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a6 + 208) + 24LL) + 8 * v10);
  v12 = *v11;
  if ( !*v11 )
  {
    v77 = sub_22077B0(0xA0u);
    *(_DWORD *)(v77 + 8) = 0;
    *(_QWORD *)(v77 + 16) = 0;
    *(_QWORD *)(v77 + 24) = 0;
    *(_QWORD *)v77 = off_4A048A0;
    *(_BYTE *)(v77 + 32) = 0;
    *(_QWORD *)(v77 + 36) = 0;
    *(_QWORD *)(v77 + 48) = 0;
    *(_QWORD *)(v77 + 56) = 0;
    *(_QWORD *)(v77 + 64) = 0;
    *(_QWORD *)(v77 + 72) = 0;
    *(_QWORD *)(v77 + 80) = 0;
    *(_QWORD *)(v77 + 88) = 0;
    *(_QWORD *)(v77 + 96) = 0;
    *(_DWORD *)(v77 + 104) = 0;
    *(_BYTE *)(v77 + 152) = 0;
    sub_2243C60(v77, v8);
    sub_2209690(*(_QWORD *)(a6 + 208), (volatile signed __int32 *)v77, v10);
    v12 = *v11;
  }
  if ( *(_QWORD *)(v12 + 72) )
    v90 = *(_QWORD *)(v12 + 88) != 0;
  else
    v90 = 0;
  v13 = *(_BYTE *)(v12 + 32) == 0;
  v96 = (volatile signed __int32 *)&unk_4FD67D8;
  if ( !v13 )
    sub_2215AB0((__int64 *)&v96, 0x20u);
  v97[0] = (volatile signed __int32 *)&unk_4FD67D8;
  sub_2215AB0((__int64 *)v97, 0x20u);
  v88 = 0;
  v14 = 0;
  v82 = 0;
  v95 = *(_DWORD *)(v12 + 104);
  v87 = 0;
  v83 = 0;
  v89 = 0;
  while ( 2 )
  {
    switch ( *((_BYTE *)&v95 + v82) )
    {
      case 0:
        LOBYTE(v15) = 1;
        goto LABEL_10;
      case 1:
        if ( sub_2247850((__int64)&v93, (__int64)&v91)
          || (v70 = sub_2247910((__int64)&v93),
              LOBYTE(v15) = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v84 + 16LL))(
                              v84,
                              0x2000,
                              v70),
              !(_BYTE)v15) )
        {
          LOBYTE(v15) = 0;
          if ( v82 == 3 )
            goto LABEL_11;
          goto LABEL_82;
        }
        sub_2240940(v93);
        c[0] = -1;
LABEL_10:
        if ( v82 != 3 )
        {
LABEL_82:
          v46 = c[0];
          v47 = v93;
          while ( 1 )
          {
            v51 = v46 == -1;
            v52 = v51 & (v47 != 0);
            if ( v52 )
            {
              v53 = (_DWORD *)v47[2];
              v54 = (unsigned __int64)v53 >= v47[3] ? (*(__int64 (__fastcall **)(_QWORD *))(*v47 + 72LL))(v47) : *v53;
              v51 = 0;
              if ( v54 == -1 )
              {
                v93 = 0;
                v51 = v52;
              }
            }
            v48 = (_DWORD)v92 == -1;
            v49 = v48 & (v91 != 0);
            if ( v49 )
            {
              v66 = (_DWORD *)v91[2];
              v67 = (unsigned __int64)v66 >= v91[3] ? (*(__int64 (**)(void))(*v91 + 72LL))() : *v66;
              v48 = 0;
              if ( v67 == -1 )
              {
                v91 = 0;
                v48 = v49;
              }
            }
            if ( v51 == v48 )
              break;
            if ( c[0] == -1 && v93 )
            {
              v68 = (_DWORD *)v93[2];
              v69 = (unsigned __int64)v68 >= v93[3] ? (*(__int64 (**)(void))(*v93 + 72LL))() : *v68;
              if ( v69 == -1 )
                v93 = 0;
            }
            if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v84 + 16LL))(v84, 0x2000) )
              break;
            v47 = v93;
            v50 = v93[2];
            if ( v50 >= v93[3] )
            {
              (*(void (__fastcall **)(_QWORD *))(*v93 + 80LL))(v93);
              v47 = v93;
            }
            else
            {
              v93[2] = v50 + 4;
            }
            c[0] = -1;
            v46 = -1;
          }
LABEL_118:
          v41 = v15 ^ 1;
LABEL_119:
          if ( (int)v82 + 1 > 3 || v41 )
            goto LABEL_11;
LABEL_121:
          ++v82;
          continue;
        }
LABEL_11:
        if ( ((unsigned __int8)v15 & (v83 > 1)) == 0 )
        {
          if ( (_BYTE)v15 )
          {
            if ( *((_QWORD *)v97[0] - 3) > 1u )
              goto LABEL_166;
            goto LABEL_148;
          }
          goto LABEL_102;
        }
        if ( v89 )
          s = *(wchar_t **)(v12 + 80);
        else
          s = *(wchar_t **)(v12 + 64);
        v16 = c[0];
        v17 = v93;
        v18 = 1;
        while ( 2 )
        {
          v23 = v16 == -1;
          v24 = v23 & (v17 != 0);
          if ( v24 )
          {
            v25 = (_DWORD *)v17[2];
            v26 = (unsigned __int64)v25 >= v17[3] ? (*(__int64 (__fastcall **)(_QWORD *))(*v17 + 72LL))(v17) : *v25;
            v23 = 0;
            if ( v26 == -1 )
            {
              v93 = 0;
              v23 = v24;
            }
          }
          v19 = (_DWORD)v92 == -1;
          v20 = v19 & (v91 != 0);
          if ( !v20
            || ((v71 = (int *)v91[2], (unsigned __int64)v71 >= v91[3])
              ? (v72 = (*(__int64 (**)(void))(*v91 + 72LL))())
              : (v72 = *v71),
                v19 = 0,
                v72 != -1) )
          {
            if ( v83 <= v18 )
              break;
            goto LABEL_17;
          }
          v91 = 0;
          v19 = v20;
          if ( v83 > v18 )
          {
LABEL_17:
            if ( v23 == v19 )
              break;
            v21 = c[0];
            if ( v93 && c[0] == -1 )
            {
              v74 = (_DWORD *)v93[2];
              v21 = (unsigned __int64)v74 >= v93[3] ? (*(__int64 (**)(void))(*v93 + 72LL))() : *v74;
              if ( v21 == -1 )
                v93 = 0;
            }
            if ( s[v18] != v21 )
              goto LABEL_102;
            v17 = v93;
            v22 = v93[2];
            if ( v22 >= v93[3] )
            {
              (*(void (__fastcall **)(_QWORD *))(*v93 + 80LL))(v93);
              v17 = v93;
            }
            else
            {
              v93[2] = v22 + 4;
            }
            c[0] = -1;
            ++v18;
            v16 = -1;
            continue;
          }
          break;
        }
        if ( v83 != v18 )
          goto LABEL_102;
        if ( *((_QWORD *)v97[0] - 3) <= 1u )
          goto LABEL_148;
LABEL_166:
        v75 = sub_22153C0(v97, 48, 0);
        if ( v75 )
        {
          v76 = *((_QWORD *)v97[0] - 3);
          if ( v75 == -1 )
            v75 = v76 - 1;
          if ( v75 <= v76 )
            v76 = v75;
          sub_2215540(v97, 0, v76, 0);
        }
LABEL_148:
        if ( v89 )
        {
          v78 = v97[0];
          if ( *((int *)v97[0] - 2) >= 0 )
          {
            sub_2215730(v97);
            v78 = v97[0];
          }
          if ( *(_BYTE *)v78 != 48 )
          {
            if ( *((int *)v78 - 2) >= 0 )
              sub_2215730(v97);
            sub_22157B0(v97, 0, 0, 1u, 45);
            *((_DWORD *)v97[0] - 2) = -1;
          }
        }
        if ( *((_QWORD *)v96 - 3) )
        {
          v73 = v14;
          if ( v88 )
            v73 = v87;
          sub_2215DF0((__int64 *)&v96, v73);
          if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v12 + 16), *(_QWORD *)(v12 + 24), &v96) )
            *a7 |= 4u;
        }
        if ( v88 && *(_DWORD *)(v12 + 96) != v14 )
        {
LABEL_102:
          *a7 |= 4u;
          v57 = sub_2247850((__int64)&v93, (__int64)&v91);
          goto LABEL_103;
        }
        sub_2215390(a8, (__int64 *)v97);
        v57 = sub_2247850((__int64)&v93, (__int64)&v91);
LABEL_103:
        if ( v57 )
          *a7 |= 2u;
        v58 = v93;
        v59 = v97[0] - 6;
        if ( v97[0] - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
        {
          if ( &_pthread_key_create )
          {
            v79 = _InterlockedExchangeAdd(v97[0] - 2, 0xFFFFFFFF);
          }
          else
          {
            v79 = *((_DWORD *)v97[0] - 2);
            *((_DWORD *)v97[0] - 2) = v79 - 1;
          }
          if ( v79 <= 0 )
            j_j___libc_free_0_1((unsigned __int64)v59);
        }
        v60 = (unsigned __int64)(v96 - 6);
        if ( v96 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
        {
          if ( &_pthread_key_create )
          {
            v80 = _InterlockedExchangeAdd(v96 - 2, 0xFFFFFFFF);
          }
          else
          {
            v80 = *((_DWORD *)v96 - 2);
            *((_DWORD *)v96 - 2) = v80 - 1;
          }
          if ( v80 <= 0 )
            j_j___libc_free_0_1(v60);
        }
        return v58;
      case 2:
        if ( (*(_BYTE *)(a6 + 25) & 2) != 0 )
          goto LABEL_66;
        v41 = (_DWORD)v82 == 0 || v83 > 1;
        if ( v41 )
          goto LABEL_66;
        if ( (_DWORD)v82 == 1 )
        {
          if ( !v90 && (_BYTE)v95 != 3 && BYTE2(v95) != 1 )
            goto LABEL_165;
        }
        else
        {
          LOBYTE(v15) = 1;
          if ( (_DWORD)v82 != 2 )
            goto LABEL_119;
          if ( HIBYTE(v95) != 4 && (HIBYTE(v95) != 3 || !v90) )
            goto LABEL_121;
        }
LABEL_66:
        v42 = 0;
        v85 = *(_QWORD *)(v12 + 56);
        while ( 1 )
        {
          v41 = !sub_2247850((__int64)&v93, (__int64)&v91) && v42 < v85;
          if ( !v41 )
            break;
          v44 = c[0];
          if ( c[0] == -1
            && v93
            && ((v45 = (wchar_t *)v93[2], (unsigned __int64)v45 >= v93[3])
              ? (v44 = (*(__int64 (**)(void))(*v93 + 72LL))())
              : (v44 = *v45),
                v44 == -1) )
          {
            v93 = 0;
            if ( *(_DWORD *)(*(_QWORD *)(v12 + 48) + 4 * v42) != -1 )
              goto LABEL_78;
          }
          else if ( *(_DWORD *)(*(_QWORD *)(v12 + 48) + 4 * v42) != v44 )
          {
            goto LABEL_78;
          }
          v43 = v93[2];
          if ( v43 >= v93[3] )
            (*(void (__fastcall **)(_QWORD *))(*v93 + 80LL))(v93);
          else
            v93[2] = v43 + 4;
          c[0] = -1;
          ++v42;
        }
        if ( v42 == v85 )
          goto LABEL_165;
LABEL_78:
        if ( v42 )
          goto LABEL_102;
        v15 = (*(_BYTE *)(a6 + 25) & 2) == 0;
        v41 = (*(_BYTE *)(a6 + 25) & 2) != 0;
        goto LABEL_119;
      case 3:
        if ( *(_QWORD *)(v12 + 72) )
        {
          v41 = sub_2247850((__int64)&v93, (__int64)&v91);
          if ( !v41 && **(_DWORD **)(v12 + 64) == (unsigned int)sub_2247910((__int64)&v93) )
          {
            v83 = *(_QWORD *)(v12 + 72);
            sub_2240940(v93);
            c[0] = -1;
LABEL_165:
            LOBYTE(v15) = 1;
            goto LABEL_119;
          }
          if ( !*(_QWORD *)(v12 + 88) )
          {
            if ( !*(_QWORD *)(v12 + 72) )
              goto LABEL_58;
LABEL_135:
            v89 = 1;
            v41 = 0;
            LOBYTE(v15) = 1;
            goto LABEL_119;
          }
        }
        else if ( !*(_QWORD *)(v12 + 88) )
        {
          goto LABEL_58;
        }
        v41 = sub_2247850((__int64)&v93, (__int64)&v91);
        if ( v41 || **(_DWORD **)(v12 + 80) != (unsigned int)sub_2247910((__int64)&v93) )
        {
          if ( !*(_QWORD *)(v12 + 72) || *(_QWORD *)(v12 + 88) )
          {
LABEL_58:
            v41 = v90;
            LOBYTE(v15) = !v90;
            goto LABEL_119;
          }
          goto LABEL_135;
        }
        v83 = *(_QWORD *)(v12 + 88);
        sub_2240940(v93);
        c[0] = -1;
        LOBYTE(v15) = 1;
        v89 = 1;
        goto LABEL_119;
      case 4:
        v27 = c[0];
        v28 = v93;
        while ( 2 )
        {
          v33 = v27 == -1;
          v34 = v33 & (v28 != 0);
          if ( v34 )
          {
            v62 = (_DWORD *)v28[2];
            v63 = (unsigned __int64)v62 >= v28[3] ? (*(__int64 (__fastcall **)(_QWORD *))(*v28 + 72LL))(v28) : *v62;
            v33 = 0;
            if ( v63 == -1 )
            {
              v93 = 0;
              v33 = v34;
            }
          }
          v35 = (_DWORD)v92 == -1;
          v36 = v35 & (v91 != 0);
          if ( v36
            && ((v55 = (int *)v91[2], (unsigned __int64)v55 >= v91[3])
              ? (v56 = (*(__int64 (**)(void))(*v91 + 72LL))())
              : (v56 = *v55),
                v35 = 0,
                v56 == -1) )
          {
            v91 = 0;
            if ( v33 == v36 )
              goto LABEL_100;
          }
          else if ( v33 == v35 )
          {
LABEL_100:
            LOBYTE(v15) = 1;
            goto LABEL_101;
          }
          v37 = c[0];
          if ( v93 && c[0] == -1 )
          {
            v64 = (wchar_t *)v93[2];
            if ( (unsigned __int64)v64 >= v93[3] )
            {
              v65 = (*(__int64 (**)(void))(*v93 + 72LL))();
              v37 = v65;
            }
            else
            {
              v37 = *v64;
              v65 = *v64;
            }
            if ( v65 == -1 )
              v93 = 0;
          }
          v38 = wmemchr((const wchar_t *)(v12 + 112), v37, 0xAu);
          if ( v38 )
          {
            v29 = off_4CDFAD0[((__int64)v38 - v12 - 108) >> 2];
            v30 = *((_QWORD *)v97[0] - 3);
            if ( (unsigned __int64)(v30 + 1) > *((_QWORD *)v97[0] - 2) || *((int *)v97[0] - 2) > 0 )
              sub_2215AB0((__int64 *)v97, v30 + 1);
            *((_BYTE *)v97[0] + *((_QWORD *)v97[0] - 3)) = v29;
            v31 = v97[0];
            if ( v97[0] - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
            {
              *((_DWORD *)v97[0] - 2) = 0;
              *((_QWORD *)v31 - 3) = v30 + 1;
              *((_BYTE *)v31 + v30 + 1) = 0;
            }
            ++v14;
            goto LABEL_36;
          }
          LOBYTE(v15) = v88 | (*(_DWORD *)(v12 + 36) != v37);
          if ( (_BYTE)v15 )
          {
            if ( !*(_BYTE *)(v12 + 32) )
              goto LABEL_101;
            if ( *(_DWORD *)(v12 + 40) == v37 )
            {
              if ( v88 )
              {
                LOBYTE(v15) = v88;
              }
              else
              {
                if ( v14 )
                {
                  v39 = *((_QWORD *)v96 - 3);
                  if ( (unsigned __int64)(v39 + 1) > *((_QWORD *)v96 - 2) || *((int *)v96 - 2) > 0 )
                    sub_2215AB0((__int64 *)&v96, v39 + 1);
                  *((_BYTE *)v96 + *((_QWORD *)v96 - 3)) = v14;
                  v40 = v96;
                  v14 = 0;
                  if ( v96 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
                  {
                    *((_DWORD *)v96 - 2) = 0;
                    *((_QWORD *)v40 - 3) = v39 + 1;
                    *((_BYTE *)v40 + v39 + 1) = 0;
                    v28 = v93;
                    v32 = v93[2];
                    if ( v32 >= v93[3] )
                      goto LABEL_55;
                    goto LABEL_37;
                  }
LABEL_36:
                  v28 = v93;
                  v32 = v93[2];
                  if ( v32 >= v93[3] )
                  {
LABEL_55:
                    (*(void (__fastcall **)(_QWORD *))(*v28 + 80LL))(v28);
                    v28 = v93;
                    goto LABEL_38;
                  }
LABEL_37:
                  v28[2] = v32 + 4;
LABEL_38:
                  c[0] = -1;
                  v27 = -1;
                  continue;
                }
                LOBYTE(v15) = 0;
              }
            }
            else
            {
              LOBYTE(v15) = *(_BYTE *)(v12 + 32);
            }
LABEL_101:
            if ( !*((_QWORD *)v97[0] - 3) )
              goto LABEL_102;
            goto LABEL_118;
          }
          break;
        }
        if ( *(int *)(v12 + 96) <= 0 )
        {
          v88 = 0;
          goto LABEL_100;
        }
        v87 = v14;
        v14 = 0;
        v88 = 1;
        goto LABEL_36;
      default:
        v41 = 0;
        goto LABEL_165;
    }
  }
}
