// Function: sub_10E3C50
// Address: 0x10e3c50
//
bool __fastcall sub_10E3C50(__int64 a1, __int64 a2)
{
  char v3; // al
  __int64 v4; // rdx
  _BYTE *v5; // r14
  __int64 v6; // r15
  unsigned int v7; // r13d
  bool v8; // al
  _QWORD *v9; // rax
  _BYTE *v10; // r14
  __int64 v11; // r15
  unsigned int v12; // r13d
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r13
  _BYTE *v17; // rax
  unsigned int v18; // r13d
  bool v19; // r13
  unsigned int v20; // ecx
  __int64 v21; // rax
  unsigned int v22; // ecx
  unsigned int v23; // r13d
  int v24; // eax
  __int64 v25; // r13
  _BYTE *v26; // rax
  unsigned int v27; // r13d
  bool v29; // r13
  unsigned int v30; // ecx
  __int64 v31; // rax
  unsigned int v32; // ecx
  unsigned int v33; // r13d
  int v34; // eax
  int v35; // [rsp-40h] [rbp-40h]
  int v36; // [rsp-40h] [rbp-40h]
  unsigned int v37; // [rsp-3Ch] [rbp-3Ch]
  unsigned int v38; // [rsp-3Ch] [rbp-3Ch]

  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v3 = *(_BYTE *)(a2 + 7) & 0x40;
  if ( v3 )
  {
    v4 = *(_QWORD *)(a2 - 8);
    v5 = *(_BYTE **)(v4 + 32);
    if ( *v5 != 44 )
    {
LABEL_17:
      v10 = *(_BYTE **)(v4 + 64);
      goto LABEL_14;
    }
  }
  else
  {
    v4 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v5 = *(_BYTE **)(a2 - v4 + 32);
    if ( *v5 != 44 )
    {
      v10 = *(_BYTE **)(a2 - v4 + 64);
      goto LABEL_14;
    }
  }
  v6 = *((_QWORD *)v5 - 8);
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(v6 + 24) == 0;
    else
      v8 = v7 == (unsigned int)sub_C444A0(v6 + 24);
  }
  else
  {
    v16 = *(_QWORD *)(v6 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 > 1 || *(_BYTE *)v6 > 0x15u )
      goto LABEL_45;
    v17 = sub_AD7630(*((_QWORD *)v5 - 8), 0, v4);
    if ( !v17 || *v17 != 17 )
    {
      if ( *(_BYTE *)(v16 + 8) == 17 )
      {
        v35 = *(_DWORD *)(v16 + 32);
        if ( v35 )
        {
          v19 = 0;
          v20 = 0;
          while ( 1 )
          {
            v37 = v20;
            v21 = sub_AD69F0((unsigned __int8 *)v6, v20);
            v22 = v37;
            if ( !v21 )
              break;
            if ( *(_BYTE *)v21 != 13 )
            {
              if ( *(_BYTE *)v21 != 17 )
                break;
              v23 = *(_DWORD *)(v21 + 32);
              if ( v23 <= 0x40 )
              {
                v19 = *(_QWORD *)(v21 + 24) == 0;
              }
              else
              {
                v24 = sub_C444A0(v21 + 24);
                v22 = v37;
                v19 = v23 == v24;
              }
              if ( !v19 )
                break;
            }
            v20 = v22 + 1;
            if ( v35 == v20 )
            {
              if ( v19 )
                goto LABEL_9;
              goto LABEL_44;
            }
          }
        }
      }
      goto LABEL_44;
    }
    v18 = *((_DWORD *)v17 + 8);
    if ( v18 <= 0x40 )
      v8 = *((_QWORD *)v17 + 3) == 0;
    else
      v8 = v18 == (unsigned int)sub_C444A0((__int64)(v17 + 24));
  }
  if ( !v8 )
  {
LABEL_44:
    v3 = *(_BYTE *)(a2 + 7) & 0x40;
LABEL_45:
    if ( v3 )
    {
      v10 = *(_BYTE **)(*(_QWORD *)(a2 - 8) + 64LL);
      goto LABEL_14;
    }
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    goto LABEL_17;
  }
LABEL_9:
  v9 = *(_QWORD **)(a1 + 8);
  if ( v9 )
    *v9 = v6;
  v4 = *((_QWORD *)v5 - 4);
  if ( !v4 )
    goto LABEL_44;
  **(_QWORD **)(a1 + 16) = v4;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) == 0 )
  {
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v10 = *(_BYTE **)(v4 + 64);
    if ( **(_BYTE ***)(a1 + 24) != v10 )
      goto LABEL_14;
    return 1;
  }
  v10 = *(_BYTE **)(*(_QWORD *)(a2 - 8) + 64LL);
  if ( **(_BYTE ***)(a1 + 24) == v10 )
    return 1;
LABEL_14:
  if ( *v10 != 44 )
    return 0;
  v11 = *((_QWORD *)v10 - 8);
  if ( *(_BYTE *)v11 == 17 )
  {
    v12 = *(_DWORD *)(v11 + 32);
    if ( v12 <= 0x40 )
    {
      if ( *(_QWORD *)(v11 + 24) )
        return 0;
    }
    else if ( v12 != (unsigned int)sub_C444A0(v11 + 24) )
    {
      return 0;
    }
  }
  else
  {
    v25 = *(_QWORD *)(v11 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 > 1 || *(_BYTE *)v11 > 0x15u )
      return 0;
    v26 = sub_AD7630(*((_QWORD *)v10 - 8), 0, v4);
    if ( !v26 || *v26 != 17 )
    {
      if ( *(_BYTE *)(v25 + 8) == 17 )
      {
        v36 = *(_DWORD *)(v25 + 32);
        if ( v36 )
        {
          v29 = 0;
          v30 = 0;
          while ( 1 )
          {
            v38 = v30;
            v31 = sub_AD69F0((unsigned __int8 *)v11, v30);
            if ( !v31 )
              break;
            v32 = v38;
            if ( *(_BYTE *)v31 != 13 )
            {
              if ( *(_BYTE *)v31 != 17 )
                break;
              v33 = *(_DWORD *)(v31 + 32);
              if ( v33 <= 0x40 )
              {
                v29 = *(_QWORD *)(v31 + 24) == 0;
              }
              else
              {
                v34 = sub_C444A0(v31 + 24);
                v32 = v38;
                v29 = v33 == v34;
              }
              if ( !v29 )
                break;
            }
            v30 = v32 + 1;
            if ( v36 == v30 )
            {
              if ( !v29 )
                return 0;
              goto LABEL_21;
            }
          }
        }
      }
      return 0;
    }
    v27 = *((_DWORD *)v26 + 8);
    if ( !(v27 <= 0x40 ? *((_QWORD *)v26 + 3) == 0 : v27 == (unsigned int)sub_C444A0((__int64)(v26 + 24))) )
      return 0;
  }
LABEL_21:
  v13 = *(_QWORD **)(a1 + 8);
  if ( v13 )
    *v13 = v11;
  v14 = *((_QWORD *)v10 - 4);
  if ( !v14 )
    return 0;
  **(_QWORD **)(a1 + 16) = v14;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v15 = *(_QWORD *)(a2 - 8);
  else
    v15 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  return **(_QWORD **)(a1 + 24) == *(_QWORD *)(v15 + 32);
}
