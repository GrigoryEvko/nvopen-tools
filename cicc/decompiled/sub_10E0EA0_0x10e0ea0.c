// Function: sub_10E0EA0
// Address: 0x10e0ea0
//
__int64 __fastcall sub_10E0EA0(char *a1, char a2)
{
  __int64 v2; // r13
  unsigned __int8 v3; // al
  _BYTE *v6; // rsi
  int v7; // edx
  char v8; // r14
  _BYTE *v9; // rdi
  __int64 v10; // rax
  unsigned int v11; // r14d
  int *v12; // rbx
  __int64 v13; // rsi
  int *v14; // rdi
  unsigned __int64 v15; // r12
  int *v16; // rdx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rax
  char v20; // cl
  __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned int v23; // ecx
  _QWORD *v24; // rax
  int v25; // ecx
  _QWORD v26[2]; // [rsp+0h] [rbp-F0h] BYREF
  _BYTE *v27; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+18h] [rbp-D8h]
  _BYTE v29[64]; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int64 v30; // [rsp+60h] [rbp-90h] BYREF
  char *v31; // [rsp+68h] [rbp-88h]
  __int64 v32; // [rsp+70h] [rbp-80h]
  int v33; // [rsp+78h] [rbp-78h]
  char v34; // [rsp+7Ch] [rbp-74h]
  char v35; // [rsp+80h] [rbp-70h] BYREF

  v2 = 0;
  if ( !a2 )
    return v2;
  v3 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
    return 0;
  if ( v3 == 85 )
  {
    v22 = *((_QWORD *)a1 - 4);
    if ( v22 )
    {
      if ( !*(_BYTE *)v22 && *(_QWORD *)(v22 + 24) == *((_QWORD *)a1 + 10) && *(_DWORD *)(v22 + 36) == 402 )
      {
        v2 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        if ( v2 )
          return v2;
      }
    }
  }
  if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) != 17 )
    return 0;
  if ( v3 != 92 )
    return 0;
  v2 = *((_QWORD *)a1 - 8);
  if ( !v2 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)a1 - 4);
  v7 = (unsigned __int8)*v6;
  if ( (_BYTE)v7 == 12 || v7 == 13 )
  {
    v10 = *((_QWORD *)a1 - 8);
  }
  else
  {
    if ( (unsigned __int8)(*v6 - 9) > 2u )
      return 0;
    v30 = 0;
    v28 = 0x800000000LL;
    v31 = &v35;
    v32 = 8;
    v33 = 0;
    v34 = 1;
    v27 = v29;
    v26[0] = &v30;
    v26[1] = &v27;
    v8 = sub_AA8FD0(v26, (__int64)v6);
    if ( v8 )
    {
      while ( 1 )
      {
        v9 = v27;
        if ( !(_DWORD)v28 )
          break;
        v6 = *(_BYTE **)&v27[8 * (unsigned int)v28 - 8];
        LODWORD(v28) = v28 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(v26, (__int64)v6) )
          goto LABEL_57;
      }
    }
    else
    {
LABEL_57:
      v9 = v27;
      v8 = 0;
    }
    if ( v9 != v29 )
      _libc_free(v9, v6);
    if ( !v34 )
      _libc_free(v31, v6);
    if ( !v8 )
      return 0;
    v10 = *((_QWORD *)a1 - 8);
  }
  v11 = *((_DWORD *)a1 + 20);
  if ( v11 != *(_DWORD *)(*(_QWORD *)(v10 + 8) + 32LL) )
    return 0;
  v12 = (int *)*((_QWORD *)a1 + 9);
  if ( !(unsigned __int8)sub_B4ED30(v12, v11, v11) )
    return 0;
  v13 = v11;
  sub_B48880((__int64 *)&v30, v11, 0);
  v14 = &v12[v11];
  if ( v14 == v12 )
  {
    v15 = v30;
LABEL_40:
    if ( (v15 & 1) != 0 )
    {
      if ( (~(-1LL << (v15 >> 58)) & (v15 >> 1)) == (1LL << (v15 >> 58)) - 1 )
        return v2;
      return 0;
    }
    v23 = *(_DWORD *)(v15 + 64);
    v13 = v23 >> 6;
    if ( (_DWORD)v13 )
    {
      v24 = *(_QWORD **)v15;
      while ( *v24 == -1 )
      {
        if ( (_QWORD *)(*(_QWORD *)v15 + 8LL * (unsigned int)(v13 - 1) + 8) == ++v24 )
          goto LABEL_54;
      }
    }
    else
    {
LABEL_54:
      v25 = v23 & 0x3F;
      if ( !v25 || (v13 = (unsigned int)v13, *(_QWORD *)(*(_QWORD *)v15 + 8LL * (unsigned int)v13) == (1LL << v25) - 1) )
      {
LABEL_44:
        if ( v15 )
        {
          if ( *(_QWORD *)v15 != v15 + 16 )
            _libc_free(*(_QWORD *)v15, v13);
          j_j___libc_free_0(v15, 72);
        }
        return v2;
      }
    }
LABEL_52:
    v2 = 0;
    goto LABEL_44;
  }
  v15 = v30;
  v16 = v12;
  while ( 1 )
  {
    v19 = (unsigned int)*v16;
    v20 = v15 & 1;
    if ( (_DWORD)v19 == -1 )
      break;
    if ( v20 )
    {
      v17 = v15 >> 58;
      v13 = ~(-1LL << (v15 >> 58));
      v18 = v13 & (v15 >> 1);
      if ( _bittest64((const __int64 *)&v18, v19) )
        return 0;
      v15 = 2 * ((v17 << 57) | v13 & (v18 | (1LL << v19))) + 1;
      v30 = v15;
    }
    else
    {
      v13 = *(_QWORD *)v15 + 8LL * ((unsigned int)v19 >> 6);
      v21 = *(_QWORD *)v13;
      if ( _bittest64(&v21, v19) )
        goto LABEL_52;
      *(_QWORD *)v13 = v21 | (1LL << (v19 & 0x3F));
      v15 = v30;
    }
    if ( v14 == ++v16 )
      goto LABEL_40;
  }
  v2 = 0;
  if ( !v20 )
    goto LABEL_44;
  return v2;
}
