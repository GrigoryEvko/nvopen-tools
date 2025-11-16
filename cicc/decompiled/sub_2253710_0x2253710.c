// Function: sub_2253710
// Address: 0x2253710
//
__int64 __fastcall sub_2253710(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        _QWORD *a4,
        _QWORD *a5,
        __int64 a6,
        _QWORD *a7,
        __int64 *a8)
{
  const char *v11; // rbp
  const char *v12; // rsi
  const char *v14; // rsi
  unsigned __int64 v15; // rax
  _QWORD *v16; // r10
  __int64 v17; // r15
  __int64 v18; // rbp
  int v19; // r14d
  int v20; // r12d
  int v21; // r9d
  int v22; // r11d
  __int64 v23; // rdx
  int v24; // r9d
  int v25; // eax
  int v26; // r8d
  int v27; // edi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r8
  char *v31; // r8
  unsigned __int8 v32; // al
  unsigned __int8 v33; // cl
  int v34; // eax
  int v35; // edx
  __int64 v36; // rdi
  int v37; // eax
  int v38; // eax
  int v39; // eax
  __int64 v40; // rax
  _QWORD *v41; // [rsp+0h] [rbp-88h]
  int v42; // [rsp+0h] [rbp-88h]
  unsigned __int64 v43; // [rsp+8h] [rbp-80h]
  unsigned __int8 v47; // [rsp+25h] [rbp-63h]
  char v48; // [rsp+26h] [rbp-62h]
  char v49; // [rsp+27h] [rbp-61h]
  _QWORD *v50; // [rsp+28h] [rbp-60h]
  __int64 v51; // [rsp+30h] [rbp-58h] BYREF
  __int64 v52; // [rsp+38h] [rbp-50h]
  __int64 v53; // [rsp+40h] [rbp-48h]

  if ( (*((_BYTE *)a8 + 20) & 0x10) != 0 )
    *((_DWORD *)a8 + 5) = *(_DWORD *)(a1 + 16);
  v11 = *(const char **)(a1 + 8);
  if ( a5 != a7 )
  {
    v12 = (const char *)a4[1];
    if ( v12 == v11 )
      goto LABEL_7;
    if ( *v11 != 42 )
    {
LABEL_6:
      if ( !strcmp(v11, v12) )
        goto LABEL_7;
    }
LABEL_13:
    v15 = 0;
    v47 = 0;
    v16 = a5;
    v49 = 0;
    v17 = a2;
    v48 = 1;
    if ( a2 >= 0 )
      v15 = (unsigned __int64)a7 - a2;
    v43 = v15;
    while ( 1 )
    {
      v18 = *(unsigned int *)(a1 + 20);
      if ( *(_DWORD *)(a1 + 20) )
        break;
LABEL_66:
      v49 &= v48;
      v48 = 0;
      if ( !v49 )
        return v47;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v27 = *((_DWORD *)a8 + 5);
        v53 = 0;
        v51 = 0;
        v28 = *(_QWORD *)(a1 + 16 * v18 + 16);
        HIDWORD(v53) = v27;
        v29 = a3;
        v52 = 0;
        v30 = v28 >> 8;
        if ( (v28 & 1) != 0 )
        {
          v29 = a3 | 1;
          v30 = *(_QWORD *)(*v16 + v30);
        }
        v31 = (char *)v16 + v30;
        if ( !v43 || v48 != v43 < (unsigned __int64)v31 )
          break;
        v49 = 1;
LABEL_38:
        if ( !--v18 )
          goto LABEL_66;
      }
      if ( (v28 & 2) != 0 )
        goto LABEL_46;
      if ( v17 != -2 || (v27 & 3) != 0 )
        break;
      if ( !--v18 )
        goto LABEL_66;
    }
    v29 = (unsigned int)v29 & 0xFFFFFFFD;
LABEL_46:
    v41 = v16;
    v32 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *, char *, __int64, _QWORD *, __int64 *))(**(_QWORD **)(a1 + 16 * v18 + 8) + 56LL))(
            *(_QWORD *)(a1 + 16 * v18 + 8),
            v17,
            v29,
            a4,
            v31,
            a6,
            a7,
            &v51);
    v20 = v53;
    v33 = v32;
    v34 = HIDWORD(v52) | *((_DWORD *)a8 + 3);
    v35 = v53;
    *((_DWORD *)a8 + 3) = v34;
    v26 = v34;
    v16 = v41;
    if ( (v35 & 0xFFFFFFFB) == 2 )
    {
      v40 = v51;
      *((_DWORD *)a8 + 4) = v20;
      v47 = v33;
      *a8 = v40;
      *((_DWORD *)a8 + 2) = v52;
      return v47;
    }
    v36 = *a8;
    v23 = v51;
    if ( v47 )
    {
      if ( !v36 )
      {
        if ( !v51 )
          goto LABEL_37;
        v19 = *((_DWORD *)a8 + 4);
        if ( v34 <= 3 )
          goto LABEL_51;
LABEL_23:
        if ( (v34 & 1) == 0 || (*((_BYTE *)a8 + 20) & 2) == 0 )
        {
          if ( !v19 )
            v19 = 1;
          if ( !v20 )
            v20 = 1;
          v21 = v19;
          v22 = v20;
          goto LABEL_30;
        }
LABEL_51:
        v21 = v19;
        v22 = v20;
        if ( v19 > 0 )
          goto LABEL_52;
        if ( v20 > 3 && ((v20 & 1) == 0 || (*(_BYTE *)(a1 + 16) & 2) == 0) )
        {
          if ( (v20 ^ 1) <= 3 )
            goto LABEL_63;
          goto LABEL_32;
        }
        if ( v17 < 0 )
        {
          if ( v17 != -2 )
          {
            v39 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD *, _QWORD))(*a4 + 64LL))(
                    a4,
                    v17,
                    v36,
                    a6,
                    a7,
                    (unsigned int)v19);
            v22 = v20;
            v16 = v41;
            v19 = v39;
            v21 = v39;
LABEL_52:
            if ( v20 <= 0 )
            {
              if ( v19 > 3 && ((v19 & 1) == 0 || (*(_BYTE *)(a1 + 16) & 2) == 0) )
              {
                if ( (v19 ^ 1) > 3 )
                {
LABEL_87:
                  v24 = v21 & 2;
                  v20 = v19;
                  goto LABEL_34;
                }
LABEL_62:
                v26 = *((_DWORD *)a8 + 3);
LABEL_63:
                *a8 = 0;
                *((_DWORD *)a8 + 4) = 1;
                v47 = 1;
                goto LABEL_37;
              }
              v23 = v51;
              if ( v17 < 0 )
              {
                if ( v17 != -2 )
                {
                  v50 = v16;
                  v42 = v21;
                  v37 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD *))(*a4 + 64LL))(
                          a4,
                          v17,
                          v51,
                          a6,
                          a7);
                  v21 = v42;
                  v16 = v50;
                  v20 = v37;
                  v22 = v37;
                  if ( (v42 ^ v37) <= 3 )
                    goto LABEL_59;
                  goto LABEL_31;
                }
LABEL_86:
                if ( (v21 ^ 1) > 3 )
                  goto LABEL_87;
                goto LABEL_62;
              }
              if ( a7 != (_QWORD *)(v51 + v17) )
                goto LABEL_86;
              if ( (v19 ^ 6) <= 3 )
              {
                v22 = 6;
LABEL_59:
                if ( (v22 & v21) > 3 )
                {
                  *a8 = 0;
                  *((_DWORD *)a8 + 4) = 2;
                  return 1;
                }
                goto LABEL_62;
              }
LABEL_106:
              v24 = 2;
              v20 = 6;
              goto LABEL_33;
            }
            goto LABEL_30;
          }
          if ( v20 <= 0 )
          {
LABEL_85:
            v19 = 1;
            v21 = 1;
            goto LABEL_86;
          }
LABEL_97:
          v19 = 1;
          if ( (v20 ^ 1) <= 3 )
            goto LABEL_63;
LABEL_31:
          v23 = v51;
          if ( v20 > 3 )
          {
LABEL_32:
            v24 = v20 & 2;
LABEL_33:
            v25 = v52;
            *a8 = v23;
            v47 = 0;
            *((_DWORD *)a8 + 2) = v25;
            goto LABEL_34;
          }
          v20 = v19;
          v24 = v19 & 2;
LABEL_34:
          *((_DWORD *)a8 + 4) = v20;
          if ( v24 || (v20 & 1) == 0 )
            return 0;
          v26 = *((_DWORD *)a8 + 3);
          goto LABEL_37;
        }
        if ( a7 != (_QWORD *)(v17 + v36) )
        {
          if ( v20 <= 0 )
          {
            if ( a7 == (_QWORD *)(v51 + v17) )
              goto LABEL_106;
            goto LABEL_85;
          }
          goto LABEL_97;
        }
        v21 = 6;
        v19 = 6;
        if ( v20 <= 0 )
          goto LABEL_87;
LABEL_30:
        if ( (v21 ^ v20) <= 3 )
          goto LABEL_59;
        goto LABEL_31;
      }
    }
    else if ( !v36 )
    {
      v38 = v52;
      *a8 = v51;
      *((_DWORD *)a8 + 2) = v38;
      if ( v23 && v26 )
      {
        v47 = v33;
        if ( (*(_BYTE *)(a1 + 16) & 1) == 0 )
          return v47;
      }
      else
      {
        v47 = v33;
      }
      goto LABEL_37;
    }
    if ( v36 == v51 )
    {
      *((_DWORD *)a8 + 2) |= v52;
    }
    else if ( v51 || v33 )
    {
      v19 = *((_DWORD *)a8 + 4);
      if ( v34 <= 3 )
        goto LABEL_51;
      goto LABEL_23;
    }
LABEL_37:
    if ( v26 == 4 )
      return v47;
    goto LABEL_38;
  }
  v14 = *(const char **)(a6 + 8);
  if ( v14 == v11 )
    goto LABEL_99;
  if ( *v11 != 42 )
  {
    if ( strcmp(*(const char **)(a1 + 8), v14) )
    {
      v12 = (const char *)a4[1];
      if ( v12 != v11 )
        goto LABEL_6;
      goto LABEL_7;
    }
LABEL_99:
    v47 = 0;
    *((_DWORD *)a8 + 3) = a3;
    return v47;
  }
  if ( (const char *)a4[1] != v11 )
    goto LABEL_13;
LABEL_7:
  *a8 = (__int64)a5;
  *((_DWORD *)a8 + 2) = a3;
  if ( a2 < 0 )
  {
    v47 = 0;
    if ( a2 == -2 )
      *((_DWORD *)a8 + 4) = 1;
  }
  else
  {
    v47 = 0;
    *((_DWORD *)a8 + 4) = 5 * (a7 == (_QWORD *)((char *)a5 + a2)) + 1;
  }
  return v47;
}
