// Function: sub_1AD4D90
// Address: 0x1ad4d90
//
__int64 __fastcall sub_1AD4D90(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rsi
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // rax
  _QWORD *v8; // r9
  unsigned __int64 v9; // rcx
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdi
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r13
  char *v19; // r12
  char *v20; // r9
  __int64 v21; // rbx
  __int64 v22; // r13
  _QWORD *v23; // r15
  char *v24; // r14
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdi
  unsigned int v28; // ecx
  __int64 v29; // rdx
  __int64 v30; // r10
  unsigned __int64 v31; // r12
  unsigned __int8 v32; // al
  unsigned __int64 v33; // rdx
  unsigned int v34; // eax
  unsigned __int64 *v35; // rdi
  unsigned __int64 v36; // rax
  bool v37; // zf
  __int64 v38; // rcx
  unsigned __int8 v39; // dl
  unsigned __int64 v40; // rsi
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // rax
  _QWORD *v43; // r12
  __int64 result; // rax
  unsigned __int64 *v45; // rbx
  unsigned __int64 v46; // rax
  __int64 *v47; // rcx
  __int64 *v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdi
  _QWORD *v53; // rax
  _QWORD *v54; // r10
  __int64 v55; // r8
  __int64 v56; // rcx
  __int64 v57; // rax
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rcx
  _QWORD *v60; // r12
  int v61; // edx
  int v62; // r8d
  __int64 v63; // rax
  unsigned __int64 *v64; // rbx
  char *v65; // r13
  unsigned __int64 v66; // rax
  __int64 v67; // [rsp+8h] [rbp-98h]
  signed __int64 v68; // [rsp+10h] [rbp-90h]
  unsigned __int64 *v69; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v71; // [rsp+28h] [rbp-78h]
  _QWORD *v72; // [rsp+30h] [rbp-70h]
  char *v73; // [rsp+38h] [rbp-68h]
  char *v74; // [rsp+38h] [rbp-68h]
  __int64 v75; // [rsp+38h] [rbp-68h]
  __int64 v76; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v77[2]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v78; // [rsp+60h] [rbp-40h]

  v5 = *a3;
  v6 = *(_QWORD *)(*(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL);
  v7 = (a1 & 0xFFFFFFFFFFFFFFF8LL) - 72;
  if ( (a1 & 4) != 0 )
    v7 = (a1 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  v8 = (_QWORD *)(v5 + 16);
  v67 = *a3;
  v72 = (_QWORD *)(v5 + 16);
  v9 = *(_QWORD *)v7;
  if ( *(_BYTE *)(*(_QWORD *)v7 + 16LL) )
    v9 = 0;
  v10 = *(_QWORD **)(v5 + 24);
  if ( v10 )
  {
    v11 = *(_QWORD **)(v5 + 24);
    do
    {
      while ( 1 )
      {
        v12 = v11[2];
        v13 = v11[3];
        if ( v11[4] >= v9 )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v13 )
          goto LABEL_10;
      }
      v8 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v12 );
LABEL_10:
    if ( v72 == v8 )
    {
      v14 = v72[5];
    }
    else
    {
      if ( v8[4] > v9 )
        v8 = v72;
      v14 = v8[5];
    }
    v15 = v72;
    do
    {
      while ( 1 )
      {
        v16 = v10[2];
        v17 = v10[3];
        if ( v10[4] >= v6 )
          break;
        v10 = (_QWORD *)v10[3];
        if ( !v17 )
          goto LABEL_18;
      }
      v15 = v10;
      v10 = (_QWORD *)v10[2];
    }
    while ( v16 );
LABEL_18:
    if ( v15 != v72 && v15[4] > v6 )
      v15 = v72;
  }
  else
  {
    v15 = (_QWORD *)(v5 + 16);
    v14 = *(_QWORD *)(v67 + 56);
  }
  v18 = v15[5];
  v19 = *(char **)(v14 + 8);
  v20 = *(char **)(v14 + 16);
  if ( v18 == v14 )
  {
    v68 = v20 - v19;
    if ( v20 == v19 )
      return sub_13983A0(v14, a1);
    if ( (unsigned __int64)(v20 - v19) > 0x7FFFFFFFFFFFFFE0LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v71 = 0;
    if ( v68 )
    {
      v74 = *(char **)(v14 + 16);
      v63 = sub_22077B0(v68);
      v20 = v74;
      v71 = (unsigned __int64 *)v63;
    }
    v64 = v71;
    if ( v19 != v20 )
    {
      v75 = v18;
      v65 = v20;
      do
      {
        if ( v64 )
        {
          *v64 = 6;
          v64[1] = 0;
          v66 = *((_QWORD *)v19 + 2);
          v64[2] = v66;
          if ( v66 != -8 && v66 != 0 && v66 != -16 )
            sub_1649AC0(v64, *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL);
          v64[3] = *((_QWORD *)v19 + 3);
        }
        v19 += 32;
        v64 += 4;
      }
      while ( v65 != v19 );
      v18 = v75;
    }
    v20 = (char *)v71 + v68;
    if ( (unsigned __int64 *)((char *)v71 + v68) == v71 )
    {
      sub_13983A0(v18, a1);
      return j_j___libc_free_0(v71, v68);
    }
    v69 = (unsigned __int64 *)((char *)v71 + v68);
    v19 = (char *)v71;
  }
  else
  {
    v68 = 0;
    v71 = 0;
    v69 = 0;
    if ( v19 == v20 )
      return sub_13983A0(v18, a1);
  }
  v73 = v20;
  v21 = v18;
  v22 = a2;
  v23 = a3;
  v24 = v19;
  do
  {
    v25 = *(unsigned int *)(v22 + 24);
    if ( !(_DWORD)v25 )
      goto LABEL_58;
    v26 = *((_QWORD *)v24 + 2);
    v27 = *(_QWORD *)(v22 + 8);
    v28 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v29 = v27 + ((unsigned __int64)v28 << 6);
    v30 = *(_QWORD *)(v29 + 24);
    if ( v26 == v30 )
    {
LABEL_26:
      if ( v29 == v27 + (v25 << 6) )
        goto LABEL_58;
      v31 = *(_QWORD *)(v29 + 56);
      if ( !v31 )
        goto LABEL_58;
      v32 = *(_BYTE *)(v31 + 16);
      if ( v32 <= 0x17u )
        goto LABEL_58;
      v33 = v31 | 4;
      if ( v32 != 78 )
      {
        if ( v32 != 29 )
          goto LABEL_31;
        v33 = v31 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v46 = v33 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v33 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_31;
      v47 = (__int64 *)(v46 - 24);
      v48 = (__int64 *)(v46 - 72);
      if ( (v33 & 4) != 0 )
        v48 = v47;
      v49 = *v48;
      if ( *(_BYTE *)(v49 + 16) || (*(_BYTE *)(v49 + 33) & 0x20) == 0 )
      {
LABEL_31:
        v77[0] = 6;
        v77[1] = 0;
        v78 = v31;
        if ( v31 != -16 && v31 != -8 )
          sub_164C220((__int64)v77);
        v34 = *((_DWORD *)v23 + 24);
        if ( v34 >= *((_DWORD *)v23 + 25) )
        {
          sub_170B450((__int64)(v23 + 11), 0);
          v34 = *((_DWORD *)v23 + 24);
        }
        v35 = (unsigned __int64 *)(v23[11] + 24LL * v34);
        if ( v35 )
        {
          *v35 = 6;
          v35[1] = 0;
          v36 = v78;
          v37 = v78 == 0;
          v35[2] = v78;
          if ( v36 != -8 && !v37 && v36 != -16 )
            sub_1649AC0(v35, v77[0] & 0xFFFFFFFFFFFFFFF8LL);
          v34 = *((_DWORD *)v23 + 24);
        }
        *((_DWORD *)v23 + 24) = v34 + 1;
        if ( v78 != -8 && v78 != 0 && v78 != -16 )
          sub_1649B30(v77);
        v38 = *((_QWORD *)v24 + 3);
        v39 = *(_BYTE *)(v31 + 16);
        v40 = *(_QWORD *)v38;
        if ( *(_QWORD *)v38 )
        {
          if ( v39 <= 0x17u )
          {
LABEL_48:
            v40 = 0;
LABEL_49:
            v41 = 0;
            v42 = 0;
          }
          else
          {
LABEL_46:
            if ( v39 == 78 )
            {
              v41 = v31 & 0xFFFFFFFFFFFFFFF8LL;
              v42 = v31 | 4;
              v40 = v31 & 0xFFFFFFFFFFFFFFF8LL;
            }
            else
            {
              if ( v39 != 29 )
                goto LABEL_48;
              v41 = v31 & 0xFFFFFFFFFFFFFFF8LL;
              v42 = v31 & 0xFFFFFFFFFFFFFFFBLL;
              v40 = v31 & 0xFFFFFFFFFFFFFFF8LL;
            }
          }
          v76 = *((_QWORD *)v24 + 3);
          v77[0] = v40;
          v43 = *(_QWORD **)(v21 + 16);
          if ( v43 == *(_QWORD **)(v21 + 24) )
          {
            sub_1398B10((char **)(v21 + 8), *(char **)(v21 + 16), v77, &v76);
            v38 = v76;
          }
          else
          {
            if ( v43 )
            {
              *v43 = 6;
              v43[1] = 0;
              v43[2] = v40;
              if ( (v42 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL && v41 )
                sub_164C220((__int64)v43);
              v38 = v76;
              v43[3] = v76;
              v43 = *(_QWORD **)(v21 + 16);
            }
            *(_QWORD *)(v21 + 16) = v43 + 4;
          }
          ++*(_DWORD *)(v38 + 32);
          goto LABEL_58;
        }
        if ( v39 <= 0x17u )
        {
          v52 = MEMORY[0xFFFFFFFFFFFFFFB8];
          if ( *(_BYTE *)(MEMORY[0xFFFFFFFFFFFFFFB8] + 16LL) )
            goto LABEL_49;
LABEL_78:
          v53 = *(_QWORD **)(v67 + 24);
          if ( v53 )
            goto LABEL_79;
          v54 = v72;
        }
        else
        {
          v50 = v31 | 4;
          if ( v39 != 78 )
          {
            v51 = 0;
            if ( v39 != 29 )
            {
LABEL_77:
              v52 = *(_QWORD *)(v51 - 72);
              if ( *(_BYTE *)(v52 + 16) )
                goto LABEL_46;
              goto LABEL_78;
            }
            v50 = v31 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v51 = v50 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v50 & 4) == 0 )
            goto LABEL_77;
          v52 = *(_QWORD *)(v51 - 24);
          if ( *(_BYTE *)(v52 + 16) )
            goto LABEL_46;
          v53 = *(_QWORD **)(v67 + 24);
          if ( !v53 )
          {
            v57 = *(_QWORD *)(v67 + 56);
            goto LABEL_87;
          }
LABEL_79:
          v54 = v72;
          do
          {
            while ( 1 )
            {
              v55 = v53[2];
              v56 = v53[3];
              if ( v53[4] >= v52 )
                break;
              v53 = (_QWORD *)v53[3];
              if ( !v56 )
                goto LABEL_83;
            }
            v54 = v53;
            v53 = (_QWORD *)v53[2];
          }
          while ( v55 );
LABEL_83:
          if ( v72 != v54 && v54[4] > v52 )
            v54 = v72;
        }
        v57 = v54[5];
        if ( v39 <= 0x17u )
        {
          v58 = 0;
          v59 = 0;
        }
        else
        {
LABEL_87:
          if ( v39 == 78 )
          {
            v58 = v31 & 0xFFFFFFFFFFFFFFF8LL;
            v59 = v31 | 4;
            v40 = v31 & 0xFFFFFFFFFFFFFFF8LL;
          }
          else
          {
            v58 = 0;
            v59 = 0;
            if ( v39 == 29 )
            {
              v58 = v31 & 0xFFFFFFFFFFFFFFF8LL;
              v59 = v31 & 0xFFFFFFFFFFFFFFFBLL;
              v40 = v31 & 0xFFFFFFFFFFFFFFF8LL;
            }
          }
        }
        v76 = v57;
        v77[0] = v40;
        v60 = *(_QWORD **)(v21 + 16);
        if ( v60 == *(_QWORD **)(v21 + 24) )
        {
          sub_1398B10((char **)(v21 + 8), *(char **)(v21 + 16), v77, &v76);
          v57 = v76;
        }
        else
        {
          if ( v60 )
          {
            *v60 = 6;
            v60[1] = 0;
            v60[2] = v40;
            if ( (v59 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL && v58 )
              sub_164C220((__int64)v60);
            v57 = v76;
            v60[3] = v76;
            v60 = *(_QWORD **)(v21 + 16);
          }
          *(_QWORD *)(v21 + 16) = v60 + 4;
        }
        ++*(_DWORD *)(v57 + 32);
      }
    }
    else
    {
      v61 = 1;
      while ( v30 != -8 )
      {
        v62 = v61 + 1;
        v28 = (v25 - 1) & (v61 + v28);
        v29 = v27 + ((unsigned __int64)v28 << 6);
        v30 = *(_QWORD *)(v29 + 24);
        if ( v26 == v30 )
          goto LABEL_26;
        v61 = v62;
      }
    }
LABEL_58:
    v24 += 32;
  }
  while ( v73 != v24 );
  sub_13983A0(v21, a1);
  result = (__int64)v71;
  if ( v69 != v71 )
  {
    v45 = v71;
    do
    {
      result = v45[2];
      if ( result != 0 && result != -8 && result != -16 )
        result = sub_1649B30(v45);
      v45 += 4;
    }
    while ( v69 != v45 );
  }
  if ( v71 )
    return j_j___libc_free_0(v71, v68);
  return result;
}
