// Function: sub_14D0990
// Address: 0x14d0990
//
__int64 *__fastcall sub_14D0990(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // r13
  int v8; // r15d
  _QWORD *v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // r14
  _QWORD *v12; // r15
  __int64 v13; // rax
  char v14; // al
  char v15; // al
  char v16; // al
  char v17; // al
  __int64 v18; // r9
  __int64 v19; // rax
  char v20; // al
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r10
  char *v24; // rax
  char *v25; // r10
  __int64 v26; // r10
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // r9
  _BYTE *v29; // r11
  int v30; // esi
  _BYTE *v31; // rcx
  __int64 v32; // rsi
  int v33; // r15d
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // rdi
  __int64 v37; // rcx
  __int64 *result; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // r9
  char v45; // dl
  char v46; // al
  __int64 v47; // rcx
  char v48; // al
  __int64 v49; // rax
  char v50; // al
  __int64 v51; // rax
  _QWORD *v52; // rdx
  int v53; // r11d
  __int64 *v54; // r10
  int v55; // edi
  int v56; // edx
  int v57; // r9d
  int v58; // r9d
  __int64 v59; // r10
  unsigned int v60; // ecx
  __int64 v61; // r8
  int v62; // edi
  __int64 *v63; // rsi
  int v64; // edi
  int v65; // edi
  __int64 v66; // r8
  __int64 *v67; // r9
  unsigned int v68; // ebx
  int v69; // ecx
  __int64 v70; // rsi
  int v71; // [rsp+Ch] [rbp-A4h]
  unsigned __int64 v72; // [rsp+10h] [rbp-A0h]
  __int64 v73; // [rsp+18h] [rbp-98h]
  char *v74; // [rsp+20h] [rbp-90h]
  unsigned __int64 v75; // [rsp+20h] [rbp-90h]
  unsigned __int64 v76; // [rsp+20h] [rbp-90h]
  __int64 v77; // [rsp+28h] [rbp-88h]
  __int64 v78; // [rsp+28h] [rbp-88h]
  __int64 v79; // [rsp+28h] [rbp-88h]
  __int64 v80; // [rsp+28h] [rbp-88h]
  __int64 v83; // [rsp+40h] [rbp-70h]
  _BYTE *v84; // [rsp+50h] [rbp-60h] BYREF
  __int64 v85; // [rsp+58h] [rbp-58h]
  _BYTE v86[80]; // [rsp+60h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a1 + 8);
  ++*(_DWORD *)(a1 + 12);
  v7 = *(_QWORD *)(a2 + 48);
  v71 = v6;
  v8 = v6;
  v83 = a2 + 40;
  if ( v7 != a2 + 40 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD **)(a4 + 16);
      v10 = *(_QWORD **)(a4 + 8);
      v11 = 0;
      if ( v7 )
        v11 = v7 - 24;
      if ( v9 == v10 )
      {
        v12 = &v10[*(unsigned int *)(a4 + 28)];
        if ( v10 == v12 )
        {
          v52 = *(_QWORD **)(a4 + 8);
        }
        else
        {
          do
          {
            if ( v11 == *v10 )
              break;
            ++v10;
          }
          while ( v12 != v10 );
          v52 = v12;
        }
        goto LABEL_58;
      }
      v12 = &v9[*(unsigned int *)(a4 + 24)];
      v10 = (_QWORD *)sub_16CC9F0(a4, v11);
      if ( v11 == *v10 )
        break;
      v13 = *(_QWORD *)(a4 + 16);
      if ( v13 == *(_QWORD *)(a4 + 8) )
      {
        v10 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a4 + 28));
        v52 = v10;
LABEL_58:
        while ( v52 != v10 && *v10 >= 0xFFFFFFFFFFFFFFFELL )
          ++v10;
        goto LABEL_10;
      }
      v10 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a4 + 24));
LABEL_10:
      if ( v10 == v12 )
      {
        v14 = *(_BYTE *)(v11 + 16);
        if ( v14 == 78 )
        {
          v43 = v11 | 4;
        }
        else
        {
          if ( v14 != 29 )
            goto LABEL_13;
          v43 = v11 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v44 = v43 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v43 & 4) != 0 )
        {
          v45 = *(_BYTE *)(*(_QWORD *)(v44 - 24) + 16LL);
          if ( !v45 )
          {
            v77 = *(_QWORD *)(v44 - 24);
            v75 = v44;
            v46 = sub_1560260(v44 + 56, 0xFFFFFFFFLL, 26);
            v47 = v77;
            if ( !v46 )
            {
              v49 = *(_QWORD *)(v75 - 24);
              if ( *(_BYTE *)(v49 + 16) )
                goto LABEL_88;
LABEL_87:
              v80 = v47;
              v84 = *(_BYTE **)(v49 + 112);
              v50 = sub_1560260(&v84, 0xFFFFFFFFLL, 26);
              v47 = v80;
              if ( !v50 )
                goto LABEL_88;
            }
            goto LABEL_80;
          }
        }
        else
        {
          v45 = *(_BYTE *)(*(_QWORD *)(v44 - 72) + 16LL);
          if ( !v45 )
          {
            v79 = *(_QWORD *)(v44 - 72);
            v76 = v44;
            v48 = sub_1560260(v44 + 56, 0xFFFFFFFFLL, 26);
            v47 = v79;
            if ( !v48 )
            {
              v49 = *(_QWORD *)(v76 - 72);
              if ( !*(_BYTE *)(v49 + 16) )
                goto LABEL_87;
LABEL_88:
              if ( (*(_BYTE *)(v47 + 32) & 0xF) == 7 )
              {
                v51 = *(_QWORD *)(v47 + 8);
                if ( v51 )
                {
                  if ( !*(_QWORD *)(v51 + 8) )
                    ++*(_DWORD *)(a1 + 60);
                }
              }
            }
LABEL_80:
            if ( *(_QWORD *)(a2 + 56) == v47 )
              *(_BYTE *)(a1 + 1) = 1;
            v78 = v47;
            if ( !sub_14A29D0(a3, (_BYTE *)v47) )
              goto LABEL_83;
            ++*(_DWORD *)(a1 + 48);
            if ( !(unsigned __int8)sub_15E4F60(v78) || (*(_BYTE *)(v78 + 33) & 0x20) != 0 )
              goto LABEL_83;
            ++*(_DWORD *)(a1 + 52);
            v14 = *(_BYTE *)(v11 + 16);
            goto LABEL_13;
          }
        }
        if ( v45 != 20 )
        {
          ++*(_DWORD *)(a1 + 48);
          ++*(_DWORD *)(a1 + 56);
LABEL_83:
          v14 = *(_BYTE *)(v11 + 16);
        }
LABEL_13:
        if ( v14 == 53 )
        {
          if ( !(unsigned __int8)sub_15F8F00(v11) )
            *(_BYTE *)(a1 + 4) = 1;
          v14 = *(_BYTE *)(v11 + 16);
        }
        if ( v14 == 83 || (v15 = *(_BYTE *)(*(_QWORD *)v11 + 8LL), v15 == 16) )
        {
          ++*(_DWORD *)(a1 + 64);
          if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 10 )
          {
LABEL_68:
            if ( (unsigned __int8)sub_15F2E00(v11, a2) )
              *(_BYTE *)(a1 + 2) = 1;
          }
        }
        else if ( v15 == 10 )
        {
          goto LABEL_68;
        }
        v16 = *(_BYTE *)(v11 + 16);
        if ( v16 == 78 )
        {
          v17 = sub_1560260(v11 + 56, 0xFFFFFFFFLL, 24);
          v18 = v11 + 56;
          if ( v17
            || (v19 = *(_QWORD *)(v11 - 24), !*(_BYTE *)(v19 + 16))
            && (v84 = *(_BYTE **)(v19 + 112), v20 = sub_1560260(&v84, 0xFFFFFFFFLL, 24), v18 = v11 + 56, v20) )
          {
            *(_BYTE *)(a1 + 2) = 1;
          }
          if ( (unsigned __int8)sub_1560260(v18, 0xFFFFFFFFLL, 8)
            || (v21 = *(_QWORD *)(v11 - 24), !*(_BYTE *)(v21 + 16))
            && (v84 = *(_BYTE **)(v21 + 112), (unsigned __int8)sub_1560260(&v84, 0xFFFFFFFFLL, 8)) )
          {
            *(_BYTE *)(a1 + 3) = 1;
          }
          v16 = *(_BYTE *)(v11 + 16);
        }
        if ( v16 == 29 )
        {
          if ( (unsigned __int8)sub_1560260(v11 + 56, 0xFFFFFFFFLL, 24)
            || (v22 = *(_QWORD *)(v11 - 72), !*(_BYTE *)(v22 + 16))
            && (v84 = *(_BYTE **)(v22 + 112), (unsigned __int8)sub_1560260(&v84, 0xFFFFFFFFLL, 24)) )
          {
            *(_BYTE *)(a1 + 2) = 1;
          }
          v16 = *(_BYTE *)(v11 + 16);
        }
        if ( v16 == 55 || v16 == 54 )
        {
          v42 = **(_QWORD **)(v11 - 24);
          if ( *(_BYTE *)(v42 + 8) == 16 )
            v42 = **(_QWORD **)(v42 + 16);
          if ( *(_DWORD *)(v42 + 8) >> 8 == 5 )
            ++*(_DWORD *)(a1 + 72);
        }
        v23 = 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
        {
          v24 = *(char **)(v11 - 8);
          v25 = &v24[v23];
        }
        else
        {
          v24 = (char *)(v11 - v23);
          v25 = (char *)v11;
        }
        v26 = v25 - v24;
        v84 = v86;
        v85 = 0x400000000LL;
        v27 = 0xAAAAAAAAAAAAAAABLL * (v26 >> 3);
        v28 = v27;
        if ( (unsigned __int64)v26 > 0x60 )
        {
          v72 = 0xAAAAAAAAAAAAAAABLL * (v26 >> 3);
          v73 = v26;
          v74 = v24;
          sub_16CD150(&v84, v86, v27, 8);
          v29 = v84;
          v30 = v85;
          LODWORD(v27) = v72;
          v24 = v74;
          v26 = v73;
          v28 = v72;
          v31 = &v84[8 * (unsigned int)v85];
        }
        else
        {
          v29 = v86;
          v30 = 0;
          v31 = v86;
        }
        if ( v26 > 0 )
        {
          do
          {
            v32 = *(_QWORD *)v24;
            v31 += 8;
            v24 += 24;
            *((_QWORD *)v31 - 1) = v32;
            --v28;
          }
          while ( v28 );
          v29 = v84;
          v30 = v85;
        }
        LODWORD(v85) = v30 + v27;
        v33 = sub_14A5330((__int64 **)a3, v11, (__int64)v29, (unsigned int)(v30 + v27));
        if ( v84 != v86 )
          _libc_free((unsigned __int64)v84);
        *(_DWORD *)(a1 + 8) += v33;
        v7 = *(_QWORD *)(v7 + 8);
        if ( v83 == v7 )
        {
LABEL_45:
          v8 = *(_DWORD *)(a1 + 8);
          goto LABEL_46;
        }
      }
      else
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v83 == v7 )
          goto LABEL_45;
      }
    }
    v40 = *(_QWORD *)(a4 + 16);
    if ( v40 == *(_QWORD *)(a4 + 8) )
      v41 = *(unsigned int *)(a4 + 28);
    else
      v41 = *(unsigned int *)(a4 + 24);
    v52 = (_QWORD *)(v40 + 8 * v41);
    goto LABEL_58;
  }
LABEL_46:
  v34 = sub_157EBA0(a2);
  if ( *(_BYTE *)(v34 + 16) == 25 )
  {
    ++*(_DWORD *)(a1 + 68);
    v34 = sub_157EBA0(a2);
  }
  v35 = *(_DWORD *)(a1 + 40);
  *(_BYTE *)(a1 + 2) |= *(_BYTE *)(v34 + 16) == 28;
  if ( !v35 )
  {
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_110;
  }
  v36 = *(_QWORD *)(a1 + 24);
  LODWORD(v37) = (v35 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v36 + 16LL * (unsigned int)v37);
  v39 = *result;
  if ( *result == a2 )
    goto LABEL_50;
  v53 = 1;
  v54 = 0;
  while ( v39 != -8 )
  {
    if ( !v54 && v39 == -16 )
      v54 = result;
    v37 = (v35 - 1) & ((_DWORD)v37 + v53);
    result = (__int64 *)(v36 + 16 * v37);
    v39 = *result;
    if ( *result == a2 )
      goto LABEL_50;
    ++v53;
  }
  v55 = *(_DWORD *)(a1 + 32);
  if ( v54 )
    result = v54;
  ++*(_QWORD *)(a1 + 16);
  v56 = v55 + 1;
  if ( 4 * (v55 + 1) >= 3 * v35 )
  {
LABEL_110:
    sub_137BFC0(a1 + 16, 2 * v35);
    v57 = *(_DWORD *)(a1 + 40);
    if ( v57 )
    {
      v58 = v57 - 1;
      v59 = *(_QWORD *)(a1 + 24);
      v56 = *(_DWORD *)(a1 + 32) + 1;
      v60 = v58 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      result = (__int64 *)(v59 + 16LL * v60);
      v61 = *result;
      if ( *result != a2 )
      {
        v62 = 1;
        v63 = 0;
        while ( v61 != -8 )
        {
          if ( !v63 && v61 == -16 )
            v63 = result;
          v60 = v58 & (v62 + v60);
          result = (__int64 *)(v59 + 16LL * v60);
          v61 = *result;
          if ( *result == a2 )
            goto LABEL_106;
          ++v62;
        }
        if ( v63 )
          result = v63;
      }
      goto LABEL_106;
    }
    goto LABEL_139;
  }
  if ( v35 - *(_DWORD *)(a1 + 36) - v56 <= v35 >> 3 )
  {
    sub_137BFC0(a1 + 16, v35);
    v64 = *(_DWORD *)(a1 + 40);
    if ( v64 )
    {
      v65 = v64 - 1;
      v66 = *(_QWORD *)(a1 + 24);
      v67 = 0;
      v68 = v65 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v56 = *(_DWORD *)(a1 + 32) + 1;
      v69 = 1;
      result = (__int64 *)(v66 + 16LL * v68);
      v70 = *result;
      if ( *result != a2 )
      {
        while ( v70 != -8 )
        {
          if ( !v67 && v70 == -16 )
            v67 = result;
          v68 = v65 & (v69 + v68);
          result = (__int64 *)(v66 + 16LL * v68);
          v70 = *result;
          if ( *result == a2 )
            goto LABEL_106;
          ++v69;
        }
        if ( v67 )
          result = v67;
      }
      goto LABEL_106;
    }
LABEL_139:
    ++*(_DWORD *)(a1 + 32);
    BUG();
  }
LABEL_106:
  *(_DWORD *)(a1 + 32) = v56;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 36);
  *((_DWORD *)result + 2) = 0;
  *result = a2;
LABEL_50:
  *((_DWORD *)result + 2) = v8 - v71;
  return result;
}
