// Function: sub_11B1B70
// Address: 0x11b1b70
//
__int64 __fastcall sub_11B1B70(__int64 a1, _BYTE *a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rax
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rdx
  _BYTE *v12; // rax
  unsigned int v13; // r14d
  char v14; // r15
  unsigned int v15; // r14d
  __int64 v16; // rax
  unsigned int v17; // r15d
  int v18; // [rsp+Ch] [rbp-34h]

  if ( a2 )
  {
    v2 = *((_QWORD *)a2 - 8);
    if ( v2 )
    {
      **(_QWORD **)a1 = v2;
      v3 = (_BYTE *)*((_QWORD *)a2 - 4);
      if ( *v3 <= 0x15u )
      {
        **(_QWORD **)(a1 + 8) = v3;
        return 1;
      }
    }
  }
  if ( *a2 != 44 )
    return 0;
  v5 = *((_QWORD *)a2 - 8);
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( v6 <= 0x40 )
      v7 = *(_QWORD *)(v5 + 24) == 0;
    else
      v7 = v6 == (unsigned int)sub_C444A0(v5 + 24);
LABEL_10:
    if ( v7 )
      goto LABEL_11;
    return 0;
  }
  v10 = *(_QWORD *)(v5 + 8);
  v11 = (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17;
  if ( (unsigned int)v11 > 1 || *(_BYTE *)v5 > 0x15u )
    return 0;
  v12 = sub_AD7630(*((_QWORD *)a2 - 8), 0, v11);
  if ( !v12 || *v12 != 17 )
  {
    if ( *(_BYTE *)(v10 + 8) == 17 )
    {
      v18 = *(_DWORD *)(v10 + 32);
      if ( v18 )
      {
        v14 = 0;
        v15 = 0;
        while ( 1 )
        {
          v16 = sub_AD69F0((unsigned __int8 *)v5, v15);
          if ( !v16 )
            break;
          if ( *(_BYTE *)v16 != 13 )
          {
            if ( *(_BYTE *)v16 != 17 )
              return 0;
            v17 = *(_DWORD *)(v16 + 32);
            if ( v17 <= 0x40 )
            {
              if ( *(_QWORD *)(v16 + 24) )
                return 0;
            }
            else if ( v17 != (unsigned int)sub_C444A0(v16 + 24) )
            {
              return 0;
            }
            v14 = 1;
          }
          if ( v18 == ++v15 )
          {
            if ( v14 )
              goto LABEL_11;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v13 = *((_DWORD *)v12 + 8);
  if ( v13 > 0x40 )
  {
    v7 = v13 == (unsigned int)sub_C444A0((__int64)(v12 + 24));
    goto LABEL_10;
  }
  if ( *((_QWORD *)v12 + 3) )
    return 0;
LABEL_11:
  v8 = *(_QWORD **)(a1 + 16);
  if ( v8 )
    *v8 = v5;
  v9 = *((_QWORD *)a2 - 4);
  if ( !v9 )
    return 0;
  **(_QWORD **)(a1 + 24) = v9;
  return 1;
}
