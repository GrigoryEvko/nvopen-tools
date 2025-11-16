// Function: sub_17255B0
// Address: 0x17255b0
//
char __fastcall sub_17255B0(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  char result; // al
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r13
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // r14d
  bool v18; // al
  __int64 v19; // rax
  __int64 *v20; // r14
  unsigned __int8 v21; // al
  unsigned int v22; // r15d
  bool v23; // al
  unsigned int v24; // r15d
  __int64 v25; // rax
  int v26; // eax
  unsigned int v27; // [rsp+8h] [rbp-38h]
  int v28; // [rsp+Ch] [rbp-34h]

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 != 39 )
  {
    if ( v4 != 5 || *(_WORD *)(a2 + 18) != 15 )
      return 0;
    v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v11 = 4 * v10;
    v12 = *(_QWORD *)(a2 - 24 * v10);
    if ( !v12 )
      goto LABEL_26;
    **a1 = v12;
    v13 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v14 = *(_BYTE *)(v13 + 16);
    if ( v14 != 37 )
    {
      if ( v14 != 5 || *(_WORD *)(v13 + 18) != 13 )
        goto LABEL_27;
      result = sub_1719130(
                 *(_BYTE **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)),
                 a2,
                 4LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF),
                 v11);
      if ( result )
      {
        v15 = *(_QWORD *)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
        if ( v15 )
        {
          *a1[2] = v15;
          return result;
        }
      }
      goto LABEL_25;
    }
    v20 = *(__int64 **)(v13 - 48);
    v21 = *((_BYTE *)v20 + 16);
    if ( v21 == 13 )
    {
      v22 = *((_DWORD *)v20 + 8);
      if ( v22 <= 0x40 )
        v23 = v20[3] == 0;
      else
        v23 = v22 == (unsigned int)sub_16A57B0((__int64)(v20 + 3));
      if ( !v23 )
        goto LABEL_27;
      v19 = *(_QWORD *)(v13 - 24);
      if ( !v19 )
        goto LABEL_27;
LABEL_33:
      *a1[2] = v19;
      return 1;
    }
    if ( *(_BYTE *)(*v20 + 8) != 16 || v21 > 0x10u )
      goto LABEL_27;
    v16 = sub_15A1020(*(_BYTE **)(v13 - 48), a2, *v20, v11);
    if ( v16 && *(_BYTE *)(v16 + 16) == 13 )
    {
      v17 = *(_DWORD *)(v16 + 32);
      if ( v17 <= 0x40 )
        v18 = *(_QWORD *)(v16 + 24) == 0;
      else
        v18 = v17 == (unsigned int)sub_16A57B0(v16 + 24);
      if ( !v18 )
        goto LABEL_25;
    }
    else
    {
      v24 = 0;
      v28 = *(_QWORD *)(*v20 + 32);
      if ( v28 )
      {
        do
        {
          v25 = sub_15A0A60((__int64)v20, v24);
          if ( !v25 )
            goto LABEL_25;
          v11 = *(unsigned __int8 *)(v25 + 16);
          if ( (_BYTE)v11 != 9 )
          {
            if ( (_BYTE)v11 != 13 )
              goto LABEL_25;
            v11 = *(unsigned int *)(v25 + 32);
            if ( (unsigned int)v11 <= 0x40 )
            {
              if ( *(_QWORD *)(v25 + 24) )
                goto LABEL_25;
            }
            else
            {
              v27 = *(_DWORD *)(v25 + 32);
              v26 = sub_16A57B0(v25 + 24);
              v11 = v27;
              if ( v27 != v26 )
                goto LABEL_25;
            }
          }
        }
        while ( v28 != ++v24 );
      }
    }
    v19 = *(_QWORD *)(v13 - 24);
    if ( v19 )
      goto LABEL_33;
LABEL_25:
    v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
LABEL_26:
    v13 = *(_QWORD *)(a2 + 24 * (1 - v10));
    if ( !v13 )
      return 0;
LABEL_27:
    **a1 = v13;
    return sub_17252E0(
             (__int64)(a1 + 1),
             *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
             4LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
             v11);
  }
  v6 = *(_QWORD *)(a2 - 48);
  if ( !v6
    || (v7 = *a1, *v7 = v6, (result = sub_171ECC0((__int64)(a1 + 1), *(_QWORD *)(a2 - 24), (__int64)v7, a4)) == 0) )
  {
    v8 = *(_QWORD *)(a2 - 24);
    if ( v8 )
    {
      v9 = *a1;
      **a1 = v8;
      return sub_171ECC0((__int64)(a1 + 1), *(_QWORD *)(a2 - 48), (__int64)v9, a4);
    }
    return 0;
  }
  return result;
}
