// Function: sub_2B266F0
// Address: 0x2b266f0
//
__int64 __fastcall sub_2B266F0(__int64 *a1, _QWORD **a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v4; // r12
  bool v5; // zf
  __int64 *v7; // r15
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 *v13; // r12
  __int64 v14; // rsi
  _QWORD *v15; // rdi
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  _BYTE *v18; // rdi
  __int64 v19; // rsi
  __int64 *v20; // r10
  _QWORD *v21; // rdx
  __int64 v22; // r8
  _QWORD *v23; // rsi
  __int64 v24; // rsi
  __int64 *v25; // rax
  char v26; // di
  __int64 v27; // r12
  __int64 v28; // rsi
  _QWORD *v29; // rax
  __int64 v30; // rcx
  _QWORD *v31; // rdx
  __int64 *v32; // rax
  __int64 *v33; // rax
  bool v34; // al
  unsigned __int8 **v35; // r8
  unsigned __int64 v36; // r9
  __int64 v37; // rsi
  _QWORD *v38; // r10
  _QWORD *v39; // rdx
  _BYTE *v40; // rsi
  unsigned __int8 **v41; // rdi
  unsigned __int8 **v42; // rax
  int v43; // edx
  int v44; // edx
  int v45; // edx
  int v46; // edx
  _BYTE *v47; // rdx
  unsigned __int8 **v48; // rax
  __int64 v49; // rdx
  unsigned __int8 **v50; // rcx
  __int64 v51; // rdx
  unsigned __int8 **v52; // rdx
  signed __int64 v53; // rdx
  signed __int64 v54; // rdx
  int v55; // edx
  unsigned __int64 v56; // rdx
  int v57; // edx
  int v58; // edx
  __int64 *v59; // [rsp+8h] [rbp-A8h]
  unsigned int v60; // [rsp+14h] [rbp-9Ch]
  _QWORD v61[2]; // [rsp+40h] [rbp-70h] BYREF
  _BYTE v62[96]; // [rsp+50h] [rbp-60h] BYREF

  LODWORD(v4) = 0;
  v61[0] = v62;
  v5 = *((_DWORD *)a2 + 26) == 3;
  v60 = a3;
  v61[1] = 0xC00000000LL;
  if ( !v5 )
    return (unsigned int)v4;
  v7 = *a2;
  v9 = *a1;
  v10 = 8LL * *((unsigned int *)a2 + 2);
  v59 = &(*a2)[(unsigned __int64)v10 / 8];
  v11 = v10 >> 3;
  v12 = v10 >> 5;
  if ( !v12 )
    goto LABEL_38;
  v13 = &v7[4 * v12];
  do
  {
    v14 = *v7;
    if ( *(_BYTE *)(v9 + 2852) )
    {
      v15 = *(_QWORD **)(v9 + 2832);
      v16 = &v15[*(unsigned int *)(v9 + 2844)];
      a4 = v15;
      if ( v15 != v16 )
      {
        v17 = *(_QWORD **)(v9 + 2832);
        while ( v14 != *v17 )
        {
          if ( v16 == ++v17 )
            goto LABEL_14;
        }
        goto LABEL_10;
      }
LABEL_14:
      v19 = v7[1];
      v20 = v7 + 1;
      goto LABEL_15;
    }
    if ( sub_C8CA60(v9 + 2824, v14) )
      goto LABEL_10;
    v19 = v7[1];
    v20 = v7 + 1;
    if ( *(_BYTE *)(v9 + 2852) )
    {
      v15 = *(_QWORD **)(v9 + 2832);
      a4 = v15;
      v16 = &v15[*(unsigned int *)(v9 + 2844)];
LABEL_15:
      if ( v16 != v15 )
      {
        v21 = v15;
        while ( *v21 != v19 )
        {
          if ( ++v21 == v16 )
            goto LABEL_20;
        }
LABEL_19:
        v7 = v20;
        goto LABEL_10;
      }
LABEL_20:
      v22 = v7[2];
      a3 = (__int64)(v7 + 2);
      goto LABEL_21;
    }
    v32 = sub_C8CA60(v9 + 2824, v19);
    v20 = v7 + 1;
    if ( v32 )
      goto LABEL_19;
    v22 = v7[2];
    a3 = (__int64)(v7 + 2);
    if ( *(_BYTE *)(v9 + 2852) )
    {
      v15 = *(_QWORD **)(v9 + 2832);
      a4 = v15;
      v16 = &v15[*(unsigned int *)(v9 + 2844)];
LABEL_21:
      if ( v15 != v16 )
      {
        v23 = v15;
        while ( *v23 != v22 )
        {
          if ( ++v23 == v16 )
            goto LABEL_29;
        }
        goto LABEL_25;
      }
LABEL_29:
      v24 = v7[3];
      a3 = (__int64)(v7 + 3);
      goto LABEL_30;
    }
    v33 = sub_C8CA60(v9 + 2824, v7[2]);
    a3 = (__int64)(v7 + 2);
    if ( v33 )
      goto LABEL_25;
    v24 = v7[3];
    a3 = (__int64)(v7 + 3);
    if ( !*(_BYTE *)(v9 + 2852) )
    {
      v25 = sub_C8CA60(v9 + 2824, v24);
      a3 = (__int64)(v7 + 3);
      if ( v25 )
        goto LABEL_25;
      goto LABEL_36;
    }
    v15 = *(_QWORD **)(v9 + 2832);
    a4 = v15;
    v16 = &v15[*(unsigned int *)(v9 + 2844)];
LABEL_30:
    if ( v16 != v15 )
    {
      while ( *a4 != v24 )
      {
        if ( ++a4 == v16 )
          goto LABEL_36;
      }
LABEL_25:
      v7 = (__int64 *)a3;
      goto LABEL_10;
    }
LABEL_36:
    v7 += 4;
  }
  while ( v13 != v7 );
  v11 = v59 - v7;
LABEL_38:
  if ( v11 == 2 )
  {
    v27 = v9 + 2824;
LABEL_62:
    v26 = *(_BYTE *)(v9 + 2852);
    v37 = *v7;
    if ( v26 )
    {
      v29 = *(_QWORD **)(v9 + 2832);
      v30 = *(unsigned int *)(v9 + 2844);
      v38 = &v29[v30];
      v39 = v29;
      if ( v29 != v38 )
      {
        while ( v37 != *v39 )
        {
          if ( v38 == ++v39 )
            goto LABEL_71;
        }
        goto LABEL_10;
      }
      v28 = v7[1];
      ++v7;
    }
    else
    {
      if ( sub_C8CA60(v27, v37) )
        goto LABEL_10;
      v26 = *(_BYTE *)(v9 + 2852);
LABEL_71:
      ++v7;
LABEL_42:
      v28 = *v7;
      if ( !v26 )
      {
        if ( !sub_C8CA60(v27, v28) )
          goto LABEL_56;
        goto LABEL_10;
      }
      v29 = *(_QWORD **)(v9 + 2832);
      v30 = *(unsigned int *)(v9 + 2844);
    }
    v31 = &v29[v30];
    if ( v29 == v31 )
      goto LABEL_56;
    while ( v28 != *v29 )
    {
      if ( v31 == ++v29 )
        goto LABEL_56;
    }
    goto LABEL_10;
  }
  if ( v11 != 3 )
  {
    if ( v11 == 1 )
    {
      v26 = *(_BYTE *)(v9 + 2852);
      v27 = v9 + 2824;
      goto LABEL_42;
    }
LABEL_56:
    v4 = &(*a2)[*((unsigned int *)a2 + 2)];
    if ( v4 == sub_2B0BF30(*a2, (__int64)v4, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
    {
      v18 = (_BYTE *)v61[0];
      LODWORD(v4) = 1;
      goto LABEL_12;
    }
    v34 = sub_2B08550((unsigned __int8 **)*a2, *((unsigned int *)a2 + 2));
    LOBYTE(v4) = v34 || (unsigned int)v36 < v60;
    if ( (_BYTE)v4 )
    {
      v18 = (_BYTE *)v61[0];
      goto LABEL_12;
    }
    v40 = a2[52];
    if ( v40 && a2[53] && *v40 == 90 )
      goto LABEL_111;
    v41 = &v35[v36];
    if ( (__int64)(8 * v36) >> 5 )
    {
      v42 = v35;
      while ( 1 )
      {
        v46 = **v42;
        if ( (_BYTE)v46 != 90 && (unsigned int)(v46 - 12) > 1 )
          break;
        v43 = *v42[1];
        if ( (_BYTE)v43 != 90 && (unsigned int)(v43 - 12) > 1 )
        {
          ++v42;
          break;
        }
        v44 = *v42[2];
        if ( (_BYTE)v44 != 90 && (unsigned int)(v44 - 12) > 1 )
        {
          v42 += 2;
          break;
        }
        v45 = *v42[3];
        if ( (_BYTE)v45 != 90 && (unsigned int)(v45 - 12) > 1 )
        {
          v42 += 3;
          break;
        }
        v42 += 4;
        if ( &v35[4 * ((__int64)(8 * v36) >> 5)] == v42 )
          goto LABEL_106;
      }
LABEL_86:
      if ( v41 != v42 )
      {
LABEL_87:
        v18 = (_BYTE *)v61[0];
        if ( v40 )
        {
          v47 = a2[53];
          if ( v47 )
          {
            LOBYTE(v42) = v47 == v40 && *v40 == 61;
            if ( (_BYTE)v42 )
            {
              LODWORD(v4) = (_DWORD)v42;
              goto LABEL_12;
            }
          }
        }
        v48 = (unsigned __int8 **)*a2;
        v49 = *((unsigned int *)a2 + 2);
        v50 = (unsigned __int8 **)&(*a2)[v49];
        v51 = (v49 * 8) >> 5;
        if ( v51 )
        {
          v52 = &v48[4 * v51];
          while ( **v48 != 61 )
          {
            if ( *v48[1] == 61 )
            {
              ++v48;
              goto LABEL_101;
            }
            if ( *v48[2] == 61 )
            {
              v48 += 2;
              goto LABEL_101;
            }
            if ( *v48[3] == 61 )
            {
              v48 += 3;
              goto LABEL_101;
            }
            v48 += 4;
            if ( v52 == v48 )
              goto LABEL_97;
          }
          goto LABEL_101;
        }
LABEL_97:
        v53 = (char *)v50 - (char *)v48;
        if ( (char *)v50 - (char *)v48 != 16 )
        {
          if ( v53 != 24 )
          {
            if ( v53 != 8 )
            {
              v48 = v50;
LABEL_101:
              LOBYTE(v4) = v50 != v48;
              goto LABEL_12;
            }
LABEL_127:
            if ( **v48 != 61 )
              goto LABEL_12;
            goto LABEL_101;
          }
          if ( **v48 == 61 )
            goto LABEL_101;
          ++v48;
        }
        if ( **v48 == 61 )
          goto LABEL_101;
        ++v48;
        goto LABEL_127;
      }
LABEL_111:
      v56 = sub_2B25EA0(v35, v36, (__int64)v61);
      v42 = (unsigned __int8 **)HIDWORD(v56);
      if ( BYTE4(v56) )
      {
        v18 = (_BYTE *)v61[0];
        LODWORD(v4) = HIDWORD(v56);
        goto LABEL_12;
      }
      v40 = a2[52];
      goto LABEL_87;
    }
    v42 = v35;
LABEL_106:
    v54 = (char *)v41 - (char *)v42;
    if ( (char *)v41 - (char *)v42 != 16 )
    {
      if ( v54 != 24 )
      {
        if ( v54 != 8 )
          goto LABEL_111;
LABEL_109:
        v55 = **v42;
        if ( (_BYTE)v55 != 90 && (unsigned int)(v55 - 12) > 1 )
          goto LABEL_86;
        goto LABEL_111;
      }
      v57 = **v42;
      if ( (_BYTE)v57 != 90 && (unsigned int)(v57 - 12) > 1 )
        goto LABEL_86;
      ++v42;
    }
    v58 = **v42;
    if ( (_BYTE)v58 != 90 && (unsigned int)(v58 - 12) > 1 )
      goto LABEL_86;
    ++v42;
    goto LABEL_109;
  }
  v27 = v9 + 2824;
  if ( !(unsigned __int8)sub_B19060(v9 + 2824, *v7, a3, (__int64)a4) )
  {
    ++v7;
    goto LABEL_62;
  }
LABEL_10:
  if ( v59 == v7 )
    goto LABEL_56;
  v18 = (_BYTE *)v61[0];
  LODWORD(v4) = 0;
LABEL_12:
  if ( v18 != v62 )
    _libc_free((unsigned __int64)v18);
  return (unsigned int)v4;
}
