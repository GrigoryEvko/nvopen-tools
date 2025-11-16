// Function: sub_9B7920
// Address: 0x9b7920
//
__int64 __fastcall sub_9B7920(char *a1)
{
  unsigned __int8 v1; // dl
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned int v7; // r14d
  bool v8; // al
  char *v9; // rdx
  __int64 v10; // rax
  char *v11; // rcx
  __int64 v12; // rax
  char *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  unsigned int v16; // ebx
  signed __int64 v17; // rax
  char v18; // r14
  unsigned int v19; // r15d
  __int64 v20; // rax
  unsigned int v21; // r14d
  int v22; // [rsp+Ch] [rbp-34h]

  v1 = *a1;
  if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a1 + 1) + 8LL) - 17 <= 1 && v1 <= 0x15u )
    return sub_AD7630(a1, 0);
  if ( v1 != 92 )
    return 0;
  v4 = *((_QWORD *)a1 - 8);
  if ( *(_BYTE *)v4 != 91 )
    return 0;
  if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
  {
    v5 = *(_QWORD *)(v4 - 8);
    v3 = *(_QWORD *)(v5 + 32);
    if ( !v3 )
      return 0;
  }
  else
  {
    v5 = v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
    v3 = *(_QWORD *)(v5 + 32);
    if ( !v3 )
      return 0;
  }
  v6 = *(_QWORD *)(v5 + 64);
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(v6 + 24) == 0;
    else
      v8 = v7 == (unsigned int)sub_C444A0(v6 + 24);
LABEL_13:
    if ( v8 )
      goto LABEL_14;
    return 0;
  }
  v14 = *(_QWORD *)(v6 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 > 1 || *(_BYTE *)v6 > 0x15u )
    return 0;
  v15 = sub_AD7630(v6, 0);
  if ( !v15 || *(_BYTE *)v15 != 17 )
  {
    if ( *(_BYTE *)(v14 + 8) == 17 )
    {
      v22 = *(_DWORD *)(v14 + 32);
      if ( v22 )
      {
        v18 = 0;
        v19 = 0;
        while ( 1 )
        {
          v20 = sub_AD69F0(v6, v19);
          if ( !v20 )
            break;
          if ( *(_BYTE *)v20 != 13 )
          {
            if ( *(_BYTE *)v20 != 17 )
              return 0;
            v21 = *(_DWORD *)(v20 + 32);
            if ( v21 <= 0x40 )
            {
              if ( *(_QWORD *)(v20 + 24) )
                return 0;
            }
            else if ( v21 != (unsigned int)sub_C444A0(v20 + 24) )
            {
              return 0;
            }
            v18 = 1;
          }
          if ( v22 == ++v19 )
          {
            if ( v18 )
              goto LABEL_14;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v16 = *(_DWORD *)(v15 + 32);
  if ( v16 > 0x40 )
  {
    v8 = v16 == (unsigned int)sub_C444A0(v15 + 24);
    goto LABEL_13;
  }
  if ( !*(_QWORD *)(v15 + 24) )
  {
LABEL_14:
    v9 = (char *)*((_QWORD *)a1 + 9);
    v10 = 4LL * *((unsigned int *)a1 + 20);
    v11 = &v9[v10];
    v12 = v10 >> 4;
    if ( v12 )
    {
      v13 = &v9[16 * v12];
      while ( (unsigned int)(*(_DWORD *)v9 + 1) <= 1 )
      {
        if ( (unsigned int)(*((_DWORD *)v9 + 1) + 1) > 1 )
        {
          v9 += 4;
          break;
        }
        if ( (unsigned int)(*((_DWORD *)v9 + 2) + 1) > 1 )
        {
          v9 += 8;
          break;
        }
        if ( (unsigned int)(*((_DWORD *)v9 + 3) + 1) > 1 )
        {
          v9 += 12;
          break;
        }
        v9 += 16;
        if ( v13 == v9 )
          goto LABEL_32;
      }
LABEL_21:
      if ( v11 == v9 )
        return v3;
      return 0;
    }
LABEL_32:
    v17 = v11 - v9;
    if ( v11 - v9 != 8 )
    {
      if ( v17 != 12 )
      {
        if ( v17 != 4 )
          return v3;
        goto LABEL_35;
      }
      if ( (unsigned int)(*(_DWORD *)v9 + 1) > 1 )
        goto LABEL_21;
      v9 += 4;
    }
    if ( (unsigned int)(*(_DWORD *)v9 + 1) > 1 )
      goto LABEL_21;
    v9 += 4;
LABEL_35:
    if ( (unsigned int)(*(_DWORD *)v9 + 1) <= 1 )
      return v3;
    goto LABEL_21;
  }
  return 0;
}
