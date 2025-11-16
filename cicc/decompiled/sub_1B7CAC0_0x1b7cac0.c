// Function: sub_1B7CAC0
// Address: 0x1b7cac0
//
__int64 __fastcall sub_1B7CAC0(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // r14
  _QWORD *v4; // rbx
  __int64 v5; // r15
  char v6; // al
  __int64 v7; // rax
  int v8; // r8d
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // rsi
  unsigned int v12; // ecx
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rax
  int v22; // r8d
  _QWORD *v23; // r14
  _QWORD *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rbx
  __int64 v29; // rsi
  __int64 v30; // rax
  bool v31; // [rsp+Bh] [rbp-45h]
  int v32; // [rsp+Ch] [rbp-44h]
  int v33; // [rsp+Ch] [rbp-44h]
  __int64 v34; // [rsp+10h] [rbp-40h] BYREF
  __int64 v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a2 - 1;
  v4 = a1 + 1;
  v5 = *a1;
  v6 = *(_BYTE *)(*a1 + 16LL);
  if ( v6 == 55 )
  {
    v15 = &v4[v3];
    v29 = (8 * v3) >> 5;
    v30 = (8 * v3) >> 3;
    if ( v29 > 0 )
    {
      while ( *(_BYTE *)(*v4 + 16LL) == 55 )
      {
        if ( *(_BYTE *)(v4[1] + 16LL) != 55 )
        {
LABEL_51:
          ++v4;
          goto LABEL_25;
        }
        if ( *(_BYTE *)(v4[2] + 16LL) != 55 )
        {
LABEL_52:
          v4 += 2;
          goto LABEL_25;
        }
        if ( *(_BYTE *)(v4[3] + 16LL) != 55 )
        {
LABEL_53:
          v4 += 3;
          goto LABEL_25;
        }
        v4 += 4;
        if ( &a1[4 * v29 + 1] == v4 )
        {
          v30 = v15 - v4;
          goto LABEL_55;
        }
      }
      goto LABEL_25;
    }
LABEL_55:
    if ( v30 != 2 )
    {
      if ( v30 != 3 )
      {
        if ( v30 != 1 )
        {
LABEL_58:
          v4 = v15;
          goto LABEL_25;
        }
LABEL_63:
        if ( *(_BYTE *)(*v4 + 16LL) == 55 )
          v4 = v15;
        goto LABEL_25;
      }
      if ( *(_BYTE *)(*v4 + 16LL) != 55 )
        goto LABEL_25;
      ++v4;
    }
    if ( *(_BYTE *)(*v4 + 16LL) != 55 )
      goto LABEL_25;
    ++v4;
    goto LABEL_63;
  }
  if ( v6 == 54 )
  {
    v15 = &v4[v3];
    v16 = (8 * v3) >> 5;
    v17 = (8 * v3) >> 3;
    if ( v16 > 0 )
    {
      while ( *(_BYTE *)(*v4 + 16LL) == 54 )
      {
        if ( *(_BYTE *)(v4[1] + 16LL) != 54 )
          goto LABEL_51;
        if ( *(_BYTE *)(v4[2] + 16LL) != 54 )
          goto LABEL_52;
        if ( *(_BYTE *)(v4[3] + 16LL) != 54 )
          goto LABEL_53;
        v4 += 4;
        if ( &a1[4 * v16 + 1] == v4 )
        {
          v17 = v15 - v4;
          goto LABEL_67;
        }
      }
      goto LABEL_25;
    }
LABEL_67:
    if ( v17 != 2 )
    {
      if ( v17 != 3 )
      {
        if ( v17 == 1 )
          goto LABEL_70;
        goto LABEL_58;
      }
      if ( *(_BYTE *)(*v4 + 16LL) != 54 )
        goto LABEL_25;
      ++v4;
    }
    if ( *(_BYTE *)(*v4 + 16LL) == 54 )
    {
      ++v4;
LABEL_70:
      if ( *(_BYTE *)(*v4 + 16LL) == 54 )
        v4 = v15;
    }
LABEL_25:
    LOBYTE(v2) = v15 == v4;
    return v2;
  }
  if ( v6 != 78 )
    BUG();
  v7 = *(_QWORD *)(v5 - 24);
  if ( *(_BYTE *)(v7 + 16) )
    BUG();
  v8 = *(_DWORD *)(v7 + 36);
  LOBYTE(v2) = v8 == 4503 || v8 == 4085;
  if ( (_BYTE)v2 )
  {
    v9 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
    v10 = *(_QWORD **)(v9 + 24);
    if ( *(_DWORD *)(v9 + 32) > 0x40u )
      v10 = (_QWORD *)*v10;
    if ( ((unsigned __int16)v10 & 0x1E0) == 0xE0 || ((unsigned __int16)v10 & 0x1C0) == 0 )
    {
      v11 = &v4[v3];
      v12 = (unsigned int)v10 & 0xFFFC1FFF;
      if ( v4 == v11 )
        return v2;
      while ( 1 )
      {
        if ( *(_BYTE *)(*v4 + 16LL) != 78 )
          BUG();
        v13 = *(_QWORD *)(*v4 - 24LL * (*(_DWORD *)(*v4 + 20LL) & 0xFFFFFFF));
        v14 = *(_QWORD **)(v13 + 24);
        if ( *(_DWORD *)(v13 + 32) > 0x40u )
          v14 = (_QWORD *)*v14;
        if ( v12 != ((unsigned int)v14 & 0xFFFC1FFF) )
          break;
        if ( v11 == ++v4 )
          return v2;
      }
    }
    return 0;
  }
  else
  {
    v31 = v8 == 4492 || v8 == 4057;
    if ( v31 )
    {
      v19 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
      v20 = *(_QWORD **)(v19 + 24);
      if ( *(_DWORD *)(v19 + 32) > 0x40u )
        v20 = (_QWORD *)*v20;
      v32 = v8;
      v21 = sub_1C278B0(v20);
      v22 = v32;
      v34 = v21;
      if ( (v21 & 0xF7) == 0 || (_BYTE)v21 == 6 )
      {
        v23 = &v4[v3];
        if ( v4 == v23 )
        {
          return v31;
        }
        else
        {
          v24 = v4;
          while ( 1 )
          {
            if ( *(_BYTE *)(*v24 + 16LL) != 78 )
              BUG();
            v25 = *(_QWORD *)(*v24 - 24LL);
            if ( *(_BYTE *)(v25 + 16) )
              BUG();
            if ( v22 != *(_DWORD *)(v25 + 36) )
              break;
            v26 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
            v27 = *(_QWORD **)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) > 0x40u )
              v27 = (_QWORD *)*v27;
            v33 = v22;
            v35[0] = sub_1C278B0(v27);
            v28 = sub_1C278C0(&v34);
            if ( v28 != sub_1C278C0(v35) )
              break;
            ++v24;
            v22 = v33;
            if ( v23 == v24 )
              return v31;
          }
        }
      }
    }
    else
    {
      return 1;
    }
  }
  return v2;
}
