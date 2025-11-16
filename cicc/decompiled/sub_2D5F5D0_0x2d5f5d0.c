// Function: sub_2D5F5D0
// Address: 0x2d5f5d0
//
__int64 __fastcall sub_2D5F5D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned __int16 v6; // ax
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 (__fastcall *v13)(__int64, int, __int16, __int64, unsigned int); // r12
  unsigned __int8 v14; // r14
  int v15; // eax
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // rdi
  const void **v19; // rsi
  unsigned int v20; // eax
  unsigned __int64 v21; // rsi
  const void *v22; // rsi
  unsigned int v23; // ecx
  unsigned __int64 v24; // r11
  bool v25; // cc
  bool v26; // al
  _BYTE *v27; // rax
  unsigned int v28; // r14d
  __int64 v29; // rdi
  _BYTE *v30; // rax
  unsigned int v31; // r14d
  bool v32; // al
  __int64 v33; // r14
  __int64 v34; // rdx
  _BYTE *v35; // rax
  unsigned int v36; // r14d
  bool v37; // al
  __int64 v38; // r14
  __int64 v39; // rdx
  _BYTE *v40; // rax
  unsigned int v41; // r14d
  bool v42; // r14
  unsigned int i; // r15d
  __int64 v44; // rax
  unsigned int v45; // r14d
  bool v46; // r15
  unsigned int j; // r14d
  __int64 v48; // rax
  unsigned int v49; // r15d
  unsigned __int64 v50; // [rsp+0h] [rbp-80h]
  unsigned int v51; // [rsp+8h] [rbp-78h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  bool v53; // [rsp+18h] [rbp-68h]
  int v54; // [rsp+18h] [rbp-68h]
  int v55; // [rsp+18h] [rbp-68h]
  _DWORD *v56; // [rsp+20h] [rbp-60h]
  const void *v58; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-48h]
  const void *v60; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v61; // [rsp+48h] [rbp-38h]

  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(_QWORD *)(a2 - 32);
  v56 = (_DWORD *)a3;
  if ( *(_BYTE *)v4 <= 0x15u && *(_BYTE *)v5 <= 0x15u )
    return 0;
  v6 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v6 == 34 )
    goto LABEL_8;
  a3 = v6;
  if ( v6 == 32 )
  {
    if ( *(_BYTE *)v5 == 17 )
    {
      v28 = *(_DWORD *)(v5 + 32);
      if ( v28 <= 0x40 )
      {
        if ( *(_QWORD *)(v5 + 24) )
          return 0;
      }
      else if ( v28 != (unsigned int)sub_C444A0(v5 + 24) )
      {
        return 0;
      }
    }
    else
    {
      v33 = *(_QWORD *)(v5 + 8);
      v34 = (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17;
      if ( (unsigned int)v34 > 1 || *(_BYTE *)v5 > 0x15u )
        return 0;
      v35 = sub_AD7630(v5, 0, v34);
      if ( v35 && *v35 == 17 )
      {
        v36 = *((_DWORD *)v35 + 8);
        if ( v36 <= 0x40 )
          v37 = *((_QWORD *)v35 + 3) == 0;
        else
          v37 = v36 == (unsigned int)sub_C444A0((__int64)(v35 + 24));
        if ( !v37 )
          return 0;
      }
      else
      {
        if ( *(_BYTE *)(v33 + 8) != 17 )
          return 0;
        v54 = *(_DWORD *)(v33 + 32);
        if ( !v54 )
          return 0;
        v42 = 0;
        for ( i = 0; i != v54; ++i )
        {
          v44 = sub_AD69F0((unsigned __int8 *)v5, i);
          if ( !v44 )
            return 0;
          if ( *(_BYTE *)v44 != 13 )
          {
            if ( *(_BYTE *)v44 != 17 )
              return 0;
            v45 = *(_DWORD *)(v44 + 32);
            v42 = v45 <= 0x40 ? *(_QWORD *)(v44 + 24) == 0 : v45 == (unsigned int)sub_C444A0(v44 + 24);
            if ( !v42 )
              return 0;
          }
        }
        if ( !v42 )
          return 0;
      }
    }
    v29 = *(_QWORD *)(v5 + 8);
    v5 = v4;
    v4 = sub_AD64C0(v29, 1, 0);
    goto LABEL_8;
  }
  if ( v6 != 33 )
  {
    if ( v6 != 36 )
      return 0;
    v4 = *(_QWORD *)(a2 - 32);
    v5 = *(_QWORD *)(a2 - 64);
    goto LABEL_8;
  }
  if ( *(_BYTE *)v5 == 17 )
  {
    v31 = *(_DWORD *)(v5 + 32);
    if ( v31 <= 0x40 )
      v32 = *(_QWORD *)(v5 + 24) == 0;
    else
      v32 = v31 == (unsigned int)sub_C444A0(v5 + 24);
    goto LABEL_61;
  }
  v38 = *(_QWORD *)(v5 + 8);
  v39 = (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17;
  if ( (unsigned int)v39 > 1 || *(_BYTE *)v5 > 0x15u )
    return 0;
  v40 = sub_AD7630(v5, 0, v39);
  if ( v40 && *v40 == 17 )
  {
    v41 = *((_DWORD *)v40 + 8);
    if ( v41 <= 0x40 )
      v32 = *((_QWORD *)v40 + 3) == 0;
    else
      v32 = v41 == (unsigned int)sub_C444A0((__int64)(v40 + 24));
LABEL_61:
    if ( !v32 )
      return 0;
    goto LABEL_8;
  }
  if ( *(_BYTE *)(v38 + 8) != 17 )
    return 0;
  v55 = *(_DWORD *)(v38 + 32);
  if ( !v55 )
    return 0;
  v46 = 0;
  for ( j = 0; j != v55; ++j )
  {
    v48 = sub_AD69F0((unsigned __int8 *)v5, j);
    if ( !v48 )
      return 0;
    if ( *(_BYTE *)v48 != 13 )
    {
      if ( *(_BYTE *)v48 != 17 )
        return 0;
      v49 = *(_DWORD *)(v48 + 32);
      v46 = v49 <= 0x40 ? *(_QWORD *)(v48 + 24) == 0 : v49 == (unsigned int)sub_C444A0(v48 + 24);
      if ( !v46 )
        return 0;
    }
  }
  if ( !v46 )
    return 0;
LABEL_8:
  v7 = v5;
  if ( *(_BYTE *)v5 < 0x16u )
    v7 = v4;
  v8 = *(_QWORD *)(v7 + 16);
  if ( !v8 )
    return 0;
  while ( 1 )
  {
    v10 = *(_QWORD *)(v8 + 24);
    if ( *(_BYTE *)v10 == 44 )
    {
      v11 = *(_QWORD *)(v10 - 64);
      if ( v5 == v11 && v11 && v4 == *(_QWORD *)(v10 - 32) )
        goto LABEL_20;
      goto LABEL_15;
    }
    if ( *(_BYTE *)v10 != 42 )
      goto LABEL_15;
    v9 = *(_QWORD *)(v10 - 64);
    if ( v5 != v9 || !v9 )
      goto LABEL_15;
    v18 = *(_QWORD *)(v10 - 32);
    if ( *(_BYTE *)v18 == 17 )
    {
      v52 = v18 + 24;
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v18 + 8) + 8LL) - 17 > 1 )
        goto LABEL_15;
      if ( *(_BYTE *)v18 > 0x15u )
        goto LABEL_15;
      v30 = sub_AD7630(v18, 0, a3);
      if ( !v30 || *v30 != 17 )
        goto LABEL_15;
      v52 = (__int64)(v30 + 24);
    }
    v19 = (const void **)(v4 + 24);
    if ( *(_BYTE *)v4 == 17 )
      break;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v4 <= 0x15u )
    {
      v27 = sub_AD7630(v4, 0, a3);
      if ( v27 )
      {
        if ( *v27 == 17 )
        {
          v19 = (const void **)(v27 + 24);
          v20 = *((_DWORD *)v27 + 8);
          v59 = v20;
          if ( v20 > 0x40 )
            goto LABEL_44;
LABEL_26:
          v21 = (unsigned __int64)*v19;
          goto LABEL_27;
        }
      }
    }
LABEL_15:
    v8 = *(_QWORD *)(v8 + 8);
    if ( !v8 )
      return 0;
  }
  v20 = *(_DWORD *)(v4 + 32);
  v59 = v20;
  if ( v20 <= 0x40 )
    goto LABEL_26;
LABEL_44:
  sub_C43780((__int64)&v58, v19);
  v20 = v59;
  if ( v59 > 0x40 )
  {
    sub_C43D10((__int64)&v58);
    goto LABEL_30;
  }
  v21 = (unsigned __int64)v58;
LABEL_27:
  v22 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & ~v21);
  if ( !v20 )
    v22 = 0;
  v58 = v22;
LABEL_30:
  sub_C46250((__int64)&v58);
  v23 = v59;
  v59 = 0;
  v24 = (unsigned __int64)v58;
  v25 = *(_DWORD *)(v52 + 8) <= 0x40u;
  v61 = v23;
  v60 = v58;
  if ( v25 )
  {
    v26 = *(_QWORD *)v52 == (_QWORD)v58;
  }
  else
  {
    v50 = (unsigned __int64)v58;
    v51 = v23;
    v26 = sub_C43C50(v52, &v60);
    v24 = v50;
    v23 = v51;
  }
  if ( v23 > 0x40 )
  {
    if ( v24 )
    {
      v53 = v26;
      j_j___libc_free_0_0(v24);
      v26 = v53;
      if ( v59 > 0x40 )
      {
        if ( v58 )
        {
          j_j___libc_free_0_0((unsigned __int64)v58);
          v26 = v53;
        }
      }
    }
  }
  if ( !v26 )
    goto LABEL_15;
LABEL_20:
  v12 = *(_QWORD *)(a1 + 16);
  v13 = *(__int64 (__fastcall **)(__int64, int, __int16, __int64, unsigned int))(*(_QWORD *)v12 + 1704LL);
  v14 = sub_BD3660(v10, 1);
  v15 = sub_2D5BAE0(*(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 816), *(__int64 **)(v10 + 8), 0);
  if ( v13 == sub_2D56A80 )
    return 0;
  if ( !(unsigned __int8)v13(v12, 79, v15, v16, v14) )
    return 0;
  result = sub_2D5EFD0(a1, v10, *(_QWORD *)(v10 - 64), *(_QWORD *)(v10 - 32), (_QWORD *)a2, 0x174u);
  if ( !(_BYTE)result )
    return 0;
  *v56 = 2;
  return result;
}
