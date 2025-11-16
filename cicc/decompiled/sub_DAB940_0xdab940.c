// Function: sub_DAB940
// Address: 0xdab940
//
__int64 __fastcall sub_DAB940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 *v8; // r13
  char v9; // di
  __int64 *v10; // r12
  __int64 *v11; // rax
  __int64 *v12; // rbx
  __int64 v13; // rcx
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r12
  __int64 *v17; // rbx
  __int64 result; // rax
  __int64 *v19; // rax
  signed __int64 v20; // r15
  __int64 *v21; // rdx
  _QWORD *v22; // rcx
  __int64 v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // r11
  __int64 *v29; // rax
  __int64 *v30; // r15
  __int64 v31; // rbx
  __int64 *v32; // r13
  __int64 *v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // rdi
  __int64 *v39; // rcx
  __int64 *v40; // rbx
  __int64 *v41; // r15
  __int64 *v42; // rax
  __int64 *v43; // rdx
  __int64 *i; // rax
  __int64 *v45; // rdi
  int v46; // edi
  int v47; // ebx
  __int64 v48; // rax
  __int64 *v49; // [rsp+8h] [rbp-E8h]
  _BYTE *v50; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v51; // [rsp+18h] [rbp-D8h]
  _BYTE v52[64]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v53; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v54; // [rsp+68h] [rbp-88h]
  __int64 v55; // [rsp+70h] [rbp-80h]
  int v56; // [rsp+78h] [rbp-78h]
  unsigned __int8 v57; // [rsp+7Ch] [rbp-74h]
  char v58; // [rsp+80h] [rbp-70h] BYREF

  v6 = (__int64 *)(a2 + 8 * a3);
  v54 = (__int64 *)&v58;
  v53 = 0;
  v55 = 8;
  v56 = 0;
  v57 = 1;
  if ( v6 == (__int64 *)a2 )
  {
LABEL_14:
    HIDWORD(v51) = 8;
    v50 = v52;
LABEL_15:
    LODWORD(v51) = 0;
    v13 = v57;
    goto LABEL_16;
  }
  v8 = (__int64 *)a2;
  v9 = 1;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        a2 = *v8;
        if ( v9 )
          break;
LABEL_26:
        ++v8;
        sub_C8CC70((__int64)&v53, a2, a3, a4, a5, a6);
        v9 = v57;
        v10 = v54;
        if ( v6 == v8 )
          goto LABEL_9;
      }
      v10 = v54;
      a4 = HIDWORD(v55);
      v11 = &v54[HIDWORD(v55)];
      if ( v54 != v11 )
        break;
LABEL_28:
      if ( HIDWORD(v55) >= (unsigned int)v55 )
        goto LABEL_26;
      a4 = (unsigned int)(HIDWORD(v55) + 1);
      ++v8;
      ++HIDWORD(v55);
      *v11 = a2;
      v10 = v54;
      ++v53;
      v9 = v57;
      if ( v6 == v8 )
        goto LABEL_9;
    }
    a3 = (__int64)v54;
    while ( a2 != *(_QWORD *)a3 )
    {
      a3 += 8;
      if ( v11 == (__int64 *)a3 )
        goto LABEL_28;
    }
    ++v8;
  }
  while ( v6 != v8 );
LABEL_9:
  if ( v9 )
    v12 = &v10[HIDWORD(v55)];
  else
    v12 = &v10[(unsigned int)v55];
  if ( v12 == v10 )
    goto LABEL_14;
  while ( (unsigned __int64)*v10 >= 0xFFFFFFFFFFFFFFFELL )
  {
    if ( ++v10 == v12 )
      goto LABEL_14;
  }
  v50 = v52;
  v51 = 0x800000000LL;
  if ( v12 == v10 )
    goto LABEL_15;
  v19 = v10;
  v20 = 0;
  while ( 1 )
  {
    v21 = v19 + 1;
    if ( v19 + 1 == v12 )
      break;
    while ( 1 )
    {
      v19 = v21;
      if ( (unsigned __int64)*v21 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v12 == ++v21 )
        goto LABEL_36;
    }
    ++v20;
    if ( v12 == v21 )
      goto LABEL_37;
  }
LABEL_36:
  ++v20;
LABEL_37:
  v22 = v52;
  if ( v20 > 8 )
  {
    a2 = (__int64)v52;
    sub_C8D5F0((__int64)&v50, v52, v20, 8u, a5, a6);
    v22 = &v50[8 * (unsigned int)v51];
  }
  v23 = *v10;
  do
  {
    v24 = v10 + 1;
    *v22++ = v23;
    if ( v12 == v10 + 1 )
      break;
    while ( 1 )
    {
      v23 = *v24;
      v10 = v24;
      if ( (unsigned __int64)*v24 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v12 == ++v24 )
        goto LABEL_43;
    }
  }
  while ( v12 != v24 );
LABEL_43:
  v13 = v57;
  LODWORD(v51) = v20 + v51;
  LODWORD(v25) = v51;
  if ( (_DWORD)v51 )
  {
    do
    {
      a2 = (unsigned int)v25;
      v25 = (unsigned int)(v25 - 1);
      a6 = *(_QWORD *)(a1 + 944);
      a5 = *(_QWORD *)&v50[8 * a2 - 8];
      v26 = *(unsigned int *)(a1 + 960);
      LODWORD(v51) = v25;
      if ( (_DWORD)v26 )
      {
        a2 = ((_DWORD)v26 - 1) & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
        v27 = a6 + 104 * a2;
        v28 = *(_QWORD *)v27;
        if ( a5 == *(_QWORD *)v27 )
        {
LABEL_48:
          a2 = 3 * v26;
          if ( v27 != a6 + 104 * v26 )
          {
            v29 = *(__int64 **)(v27 + 16);
            a2 = *(_BYTE *)(v27 + 36) ? *(unsigned int *)(v27 + 28) : *(unsigned int *)(v27 + 24);
            v30 = &v29[a2];
            if ( v29 != v30 )
            {
              while ( 1 )
              {
                v31 = *v29;
                v32 = v29;
                if ( (unsigned __int64)*v29 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v30 == ++v29 )
                  goto LABEL_45;
              }
              if ( v30 != v29 )
              {
                if ( (_BYTE)v13 )
                {
LABEL_57:
                  v33 = v54;
                  a2 = HIDWORD(v55);
                  v25 = (__int64)&v54[HIDWORD(v55)];
                  if ( v54 != (__int64 *)v25 )
                  {
                    do
                    {
                      if ( v31 == *v33 )
                        goto LABEL_61;
                      ++v33;
                    }
                    while ( (__int64 *)v25 != v33 );
                  }
                  if ( HIDWORD(v55) >= (unsigned int)v55 )
                    goto LABEL_67;
                  a2 = (unsigned int)++HIDWORD(v55);
                  *(_QWORD *)v25 = v31;
                  ++v53;
LABEL_68:
                  v35 = (unsigned int)v51;
                  v36 = (unsigned int)v51 + 1LL;
                  if ( v36 > HIDWORD(v51) )
                  {
                    a2 = (__int64)v52;
                    sub_C8D5F0((__int64)&v50, v52, v36, 8u, a5, a6);
                    v35 = (unsigned int)v51;
                  }
                  v25 = (__int64)v50;
                  *(_QWORD *)&v50[8 * v35] = v31;
                  v13 = v57;
                  LODWORD(v51) = v51 + 1;
                  goto LABEL_61;
                }
                while ( 1 )
                {
LABEL_67:
                  a2 = v31;
                  sub_C8CC70((__int64)&v53, v31, v25, v13, a5, a6);
                  v13 = v57;
                  if ( (_BYTE)v25 )
                    goto LABEL_68;
LABEL_61:
                  v34 = v32 + 1;
                  if ( v32 + 1 == v30 )
                    break;
                  while ( 1 )
                  {
                    v31 = *v34;
                    v32 = v34;
                    if ( (unsigned __int64)*v34 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( v30 == ++v34 )
                      goto LABEL_64;
                  }
                  if ( v30 == v34 )
                    break;
                  if ( (_BYTE)v13 )
                    goto LABEL_57;
                }
LABEL_64:
                LODWORD(v25) = v51;
              }
            }
          }
        }
        else
        {
          v46 = 1;
          while ( v28 != -4096 )
          {
            v47 = v46 + 1;
            a2 = ((_DWORD)v26 - 1) & (unsigned int)(v46 + a2);
            v27 = a6 + 104LL * (unsigned int)a2;
            v28 = *(_QWORD *)v27;
            if ( a5 == *(_QWORD *)v27 )
              goto LABEL_48;
            v46 = v47;
          }
        }
      }
LABEL_45:
      ;
    }
    while ( (_DWORD)v25 );
  }
LABEL_16:
  v14 = v54;
  if ( (_BYTE)v13 )
  {
    v15 = HIDWORD(v55);
    v16 = &v54[HIDWORD(v55)];
  }
  else
  {
    v15 = (unsigned int)v55;
    v16 = &v54[(unsigned int)v55];
  }
  if ( v54 == v16 )
    goto LABEL_21;
  while ( 1 )
  {
    a2 = *v14;
    v17 = v14;
    if ( (unsigned __int64)*v14 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v16 == ++v14 )
      goto LABEL_21;
  }
LABEL_78:
  if ( v17 == v16 || (sub_DAA4C0(a1, a2, v15, v13, a5, a6), v37 = v17 + 1, v17 + 1 == v16) )
  {
LABEL_21:
    result = *(unsigned int *)(a1 + 1208);
    if ( (_DWORD)result )
      goto LABEL_83;
  }
  else
  {
    do
    {
      a2 = *v37;
      v17 = v37;
      if ( (unsigned __int64)*v37 < 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_78;
      ++v37;
    }
    while ( v16 != v37 );
    result = *(unsigned int *)(a1 + 1208);
    if ( (_DWORD)result )
    {
LABEL_83:
      v38 = *(unsigned int *)(a1 + 1216);
      v39 = *(__int64 **)(a1 + 1200);
      v40 = &v39[8 * v38];
      if ( v39 != v40 )
      {
        result = *(_QWORD *)(a1 + 1200);
        while ( 1 )
        {
          a2 = *(_QWORD *)result;
          v41 = (__int64 *)result;
          if ( *(_QWORD *)result != -4096 )
            break;
          if ( *(_QWORD *)(result + 8) != -4096 )
            goto LABEL_87;
LABEL_126:
          result += 64;
          if ( v40 == (__int64 *)result )
            goto LABEL_22;
        }
        if ( a2 == -8192 && *(_QWORD *)(result + 8) == -8192 )
          goto LABEL_126;
LABEL_87:
        if ( (__int64 *)result != v40 )
        {
          while ( 2 )
          {
            if ( a2 )
              a2 += 32;
            if ( v57 )
            {
              v42 = v54;
              v43 = &v54[HIDWORD(v55)];
              if ( v54 == v43 )
              {
LABEL_112:
                v41 += 8;
                if ( v41 != v40 )
                {
                  v48 = *v41;
                  if ( *v41 == -4096 )
                    goto LABEL_118;
LABEL_114:
                  if ( v48 == -8192 && v41[1] == -8192 )
                  {
                    do
                    {
                      v41 += 8;
                      if ( v40 == v41 )
                        break;
                      v48 = *v41;
                      if ( *v41 != -4096 )
                        goto LABEL_114;
LABEL_118:
                      ;
                    }
                    while ( v41[1] == -4096 );
                  }
                }
LABEL_101:
                result = (__int64)&v39[8 * v38];
                if ( v41 == (__int64 *)result )
                  goto LABEL_22;
                a2 = *v41;
                continue;
              }
              while ( a2 != *v42 )
              {
                if ( v43 == ++v42 )
                  goto LABEL_112;
              }
            }
            else if ( !sub_C8CA60((__int64)&v53, a2) )
            {
              v39 = *(__int64 **)(a1 + 1200);
              v38 = *(unsigned int *)(a1 + 1216);
              goto LABEL_112;
            }
            break;
          }
          for ( i = v41 + 8; v40 != i; i += 8 )
          {
            if ( *i == -4096 )
            {
              if ( i[1] != -4096 )
                break;
            }
            else if ( *i != -8192 || i[1] != -8192 )
            {
              break;
            }
          }
          v45 = (__int64 *)v41[3];
          if ( v45 != v41 + 5 )
          {
            v49 = i;
            _libc_free(v45, a2);
            i = v49;
          }
          *v41 = -8192;
          v41[1] = -8192;
          v39 = *(__int64 **)(a1 + 1200);
          v41 = i;
          --*(_DWORD *)(a1 + 1208);
          v38 = *(unsigned int *)(a1 + 1216);
          ++*(_DWORD *)(a1 + 1212);
          goto LABEL_101;
        }
      }
    }
  }
LABEL_22:
  if ( v50 != v52 )
    result = _libc_free(v50, a2);
  if ( !v57 )
    return _libc_free(v54, a2);
  return result;
}
