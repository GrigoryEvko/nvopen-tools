// Function: sub_14B0710
// Address: 0x14b0710
//
bool __fastcall sub_14B0710(__int64 a1, __int64 a2, char a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  int v6; // eax
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  int v9; // eax
  __int64 *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r12
  int v13; // eax
  int v14; // eax
  _QWORD *v15; // rax
  char v17; // al
  __int64 v18; // rcx
  int v19; // eax
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  char v23; // al
  unsigned __int64 v24; // rdx
  void *v25; // rcx
  unsigned __int64 v26; // rdx
  void *v27; // rcx
  unsigned __int64 v28; // rdx
  void *v29; // rcx
  __int64 v30; // rbx
  __int64 v31; // r14
  unsigned int v32; // r15d
  bool v33; // al
  __int64 *v34; // rax
  __int64 v35; // rax
  unsigned int v36; // ebx
  bool v37; // al
  int v38; // ebx
  unsigned int v39; // r15d
  __int64 v40; // rax
  char v41; // dl
  int v42; // [rsp+Ch] [rbp-44h]
  _BYTE v43[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v44; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned __int8 *)(a1 + 16);
  if ( a3 )
  {
    if ( (unsigned __int8)v3 > 0x17u )
    {
      if ( (unsigned __int8)v3 > 0x2Fu )
        goto LABEL_4;
      v18 = 0x80A800000000LL;
      if ( !_bittest64(&v18, v3) )
        goto LABEL_4;
      v19 = (unsigned __int8)v3 - 24;
    }
    else
    {
      if ( (_BYTE)v3 != 5 )
        goto LABEL_4;
      v28 = *(unsigned __int16 *)(a1 + 18);
      if ( (unsigned __int16)v28 > 0x17u )
        goto LABEL_4;
      v29 = &loc_80A800;
      v19 = (unsigned __int16)v28;
      if ( !_bittest64((const __int64 *)&v29, v28) )
        goto LABEL_4;
    }
    if ( v19 == 13 && (*(_BYTE *)(a1 + 17) & 4) != 0 )
    {
      v20 = (__int64 *)sub_13CF970(a1);
      if ( (unsigned __int8)sub_14A95E0(*v20) )
      {
        if ( a2 == *(_QWORD *)(sub_13CF970(a1) + 24) )
          return 1;
      }
    }
LABEL_4:
    v4 = *(unsigned __int8 *)(a2 + 16);
    if ( (unsigned __int8)v4 <= 0x17u )
    {
      if ( (_BYTE)v4 == 5 )
      {
        v24 = *(unsigned __int16 *)(a2 + 18);
        if ( (unsigned __int16)v24 <= 0x17u )
        {
          v25 = &loc_80A800;
          v6 = (unsigned __int16)v24;
          if ( _bittest64((const __int64 *)&v25, v24) )
          {
LABEL_8:
            if ( v6 == 13 && (*(_BYTE *)(a2 + 17) & 4) != 0 )
            {
              v34 = (__int64 *)sub_13CF970(a2);
              if ( (unsigned __int8)sub_14A95E0(*v34) )
              {
                if ( a1 == *(_QWORD *)(sub_13CF970(a2) + 24) )
                  return 1;
              }
            }
          }
        }
      }
    }
    else if ( (unsigned __int8)v4 <= 0x2Fu )
    {
      v5 = 0x80A800000000LL;
      if ( _bittest64(&v5, v4) )
      {
        v6 = (unsigned __int8)v4 - 24;
        goto LABEL_8;
      }
    }
    v7 = *(unsigned __int8 *)(a1 + 16);
    if ( (unsigned __int8)v7 > 0x17u )
    {
      if ( (unsigned __int8)v7 <= 0x2Fu )
      {
        v8 = 0x80A800000000LL;
        if ( _bittest64(&v8, v7) )
        {
          v9 = (unsigned __int8)v7 - 24;
          goto LABEL_14;
        }
      }
      return 0;
    }
    if ( (_BYTE)v7 != 5 )
      return 0;
    v26 = *(unsigned __int16 *)(a1 + 18);
    if ( (unsigned __int16)v26 > 0x17u )
      return 0;
    v27 = &loc_80A800;
    v9 = (unsigned __int16)v26;
    if ( _bittest64((const __int64 *)&v27, v26) )
    {
LABEL_14:
      if ( v9 == 13 && (*(_BYTE *)(a1 + 17) & 4) != 0 )
      {
        v10 = (__int64 *)sub_13CF970(a1);
        v11 = *v10;
        if ( *v10 )
        {
          v12 = v10[3];
          if ( v12 )
          {
            if ( (unsigned __int8)sub_14AA490(a2) )
            {
              v13 = *(unsigned __int8 *)(a2 + 16);
              v14 = (unsigned __int8)v13 <= 0x17u ? *(unsigned __int16 *)(a2 + 18) : v13 - 24;
              if ( v14 == 13 && (*(_BYTE *)(a2 + 17) & 4) != 0 )
              {
                v15 = (_QWORD *)sub_13CF970(a2);
                if ( v12 == *v15 )
                  return v15[3] == v11;
              }
            }
          }
        }
      }
      return 0;
    }
    return 0;
  }
  if ( (_BYTE)v3 == 37 )
  {
    if ( (unsigned __int8)sub_14A95E0(*(_QWORD *)(a1 - 48)) && a2 == *(_QWORD *)(a1 - 24) )
      return 1;
  }
  else
  {
    if ( (_BYTE)v3 != 5 || *(_WORD *)(a1 + 18) != 13 )
      goto LABEL_28;
    v30 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    v31 = *(_QWORD *)(a1 - 24 * v30);
    if ( *(_BYTE *)(v31 + 16) == 13 )
    {
      v32 = *(_DWORD *)(v31 + 32);
      if ( v32 <= 0x40 )
        v33 = *(_QWORD *)(v31 + 24) == 0;
      else
        v33 = v32 == (unsigned int)sub_16A57B0(v31 + 24);
      if ( !v33 )
        goto LABEL_28;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) != 16 )
        goto LABEL_28;
      v35 = sub_15A1020(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      if ( v35 && *(_BYTE *)(v35 + 16) == 13 )
      {
        v36 = *(_DWORD *)(v35 + 32);
        if ( v36 <= 0x40 )
          v37 = *(_QWORD *)(v35 + 24) == 0;
        else
          v37 = v36 == (unsigned int)sub_16A57B0(v35 + 24);
        if ( !v37 )
          goto LABEL_28;
      }
      else
      {
        v38 = *(_QWORD *)(*(_QWORD *)v31 + 32LL);
        if ( v38 )
        {
          v39 = 0;
          do
          {
            v40 = sub_15A0A60(v31, v39);
            if ( !v40 )
              goto LABEL_28;
            v41 = *(_BYTE *)(v40 + 16);
            if ( v41 != 9 )
            {
              if ( v41 != 13 )
                goto LABEL_28;
              if ( *(_DWORD *)(v40 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v40 + 24) )
                  goto LABEL_28;
              }
              else
              {
                v42 = *(_DWORD *)(v40 + 32);
                if ( v42 != (unsigned int)sub_16A57B0(v40 + 24) )
                  goto LABEL_28;
              }
            }
          }
          while ( v38 != ++v39 );
        }
      }
      v30 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
    if ( a2 == *(_QWORD *)(a1 + 24 * (1 - v30)) )
      return 1;
  }
LABEL_28:
  v44 = a1;
  if ( sub_14B0440((__int64)v43, a2) )
    return 1;
  v17 = *(_BYTE *)(a1 + 16);
  if ( v17 == 37 )
  {
    v21 = *(_QWORD *)(a1 - 48);
    if ( !v21 )
      return 0;
    v22 = *(_QWORD *)(a1 - 24);
    if ( !v22 )
      return 0;
  }
  else
  {
    if ( v17 != 5 )
      return 0;
    if ( *(_WORD *)(a1 + 18) != 13 )
      return 0;
    v21 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( !v21 )
      return 0;
    v22 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    if ( !v22 )
      return 0;
  }
  v23 = *(_BYTE *)(a2 + 16);
  if ( v23 == 37 )
  {
    if ( v22 != *(_QWORD *)(a2 - 48) )
      return 0;
    return *(_QWORD *)(a2 - 24) == v21;
  }
  else
  {
    if ( v23 != 5 || *(_WORD *)(a2 + 18) != 13 || v22 != *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
      return 0;
    return *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == v21;
  }
}
