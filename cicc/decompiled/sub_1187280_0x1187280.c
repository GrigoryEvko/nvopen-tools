// Function: sub_1187280
// Address: 0x1187280
//
bool __fastcall sub_1187280(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // r13
  unsigned int v9; // r14d
  bool v10; // al
  __int64 v11; // r14
  __int64 v12; // rdx
  _BYTE *v13; // rax
  int v14; // r14d
  unsigned int v15; // r15d
  __int64 v16; // rax
  unsigned int v17; // esi
  __int64 v18; // r14
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned int v21; // r14d
  int v22; // r14d
  unsigned int v23; // r15d
  __int64 v24; // rax
  unsigned int v25; // esi
  unsigned int v26; // r14d
  int v27; // [rsp-3Ch] [rbp-3Ch]
  int v28; // [rsp-3Ch] [rbp-3Ch]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( !v6 )
      goto LABEL_47;
    if ( v6 <= 0x40 )
      v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) == *(_QWORD *)(v5 + 24);
    else
      v7 = v6 == (unsigned int)sub_C445E0(v5 + 24);
    goto LABEL_7;
  }
  v18 = *(_QWORD *)(v5 + 8);
  v19 = (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17;
  if ( (unsigned int)v19 > 1 || *(_BYTE *)v5 > 0x15u )
    goto LABEL_8;
  v20 = sub_AD7630(v5, 0, v19);
  if ( !v20 || *v20 != 17 )
  {
    if ( *(_BYTE *)(v18 + 8) == 17 )
    {
      v22 = *(_DWORD *)(v18 + 32);
      if ( v22 )
      {
        v23 = 0;
        while ( 1 )
        {
          v24 = sub_AD69F0((unsigned __int8 *)v5, v23);
          if ( !v24 || *(_BYTE *)v24 != 17 )
            break;
          v25 = *(_DWORD *)(v24 + 32);
          if ( v25 )
          {
            if ( v25 <= 0x40 )
            {
              if ( *(_QWORD *)(v24 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25) )
                goto LABEL_8;
            }
            else
            {
              v28 = *(_DWORD *)(v24 + 32);
              if ( v28 != (unsigned int)sub_C445E0(v24 + 24) )
                goto LABEL_8;
            }
          }
          if ( v22 == ++v23 )
            goto LABEL_47;
        }
      }
    }
    goto LABEL_8;
  }
  v21 = *((_DWORD *)v20 + 8);
  if ( v21 )
  {
    if ( v21 <= 0x40 )
      v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21) == *((_QWORD *)v20 + 3);
    else
      v7 = v21 == (unsigned int)sub_C445E0((__int64)(v20 + 24));
LABEL_7:
    if ( !v7 )
    {
LABEL_8:
      v8 = *((_QWORD *)a3 - 4);
      goto LABEL_9;
    }
  }
LABEL_47:
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v5;
  v8 = *((_QWORD *)a3 - 4);
  result = 1;
  if ( *(_QWORD *)(a1 + 8) != v8 )
  {
LABEL_9:
    if ( *(_BYTE *)v8 == 17 )
    {
      v9 = *(_DWORD *)(v8 + 32);
      if ( !v9 )
      {
LABEL_28:
        if ( *(_QWORD *)a1 )
          **(_QWORD **)a1 = v8;
        return *((_QWORD *)a3 - 8) == *(_QWORD *)(a1 + 8);
      }
      if ( v9 <= 0x40 )
        v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == *(_QWORD *)(v8 + 24);
      else
        v10 = v9 == (unsigned int)sub_C445E0(v8 + 24);
    }
    else
    {
      v11 = *(_QWORD *)(v8 + 8);
      v12 = (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17;
      if ( (unsigned int)v12 > 1 || *(_BYTE *)v8 > 0x15u )
        return 0;
      v13 = sub_AD7630(v8, 0, v12);
      if ( !v13 || *v13 != 17 )
      {
        if ( *(_BYTE *)(v11 + 8) == 17 )
        {
          v14 = *(_DWORD *)(v11 + 32);
          if ( v14 )
          {
            v15 = 0;
            while ( 1 )
            {
              v16 = sub_AD69F0((unsigned __int8 *)v8, v15);
              if ( !v16 || *(_BYTE *)v16 != 17 )
                break;
              v17 = *(_DWORD *)(v16 + 32);
              if ( v17 )
              {
                if ( v17 <= 0x40 )
                {
                  if ( *(_QWORD *)(v16 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) )
                    return 0;
                }
                else
                {
                  v27 = *(_DWORD *)(v16 + 32);
                  if ( v27 != (unsigned int)sub_C445E0(v16 + 24) )
                    return 0;
                }
              }
              if ( v14 == ++v15 )
                goto LABEL_28;
            }
          }
        }
        return 0;
      }
      v26 = *((_DWORD *)v13 + 8);
      if ( !v26 )
        goto LABEL_28;
      if ( v26 <= 0x40 )
        v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) == *((_QWORD *)v13 + 3);
      else
        v10 = v26 == (unsigned int)sub_C445E0((__int64)(v13 + 24));
    }
    if ( !v10 )
      return 0;
    goto LABEL_28;
  }
  return result;
}
