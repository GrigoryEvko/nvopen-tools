// Function: sub_8EEF90
// Address: 0x8eef90
//
__int64 __fastcall sub_8EEF90(__int64 a1, unsigned __int8 *a2)
{
  int v2; // eax
  unsigned __int8 v4; // dl
  unsigned __int8 v5; // al
  unsigned __int8 v6; // cl
  _BOOL4 v7; // edi
  unsigned __int8 v8; // al
  char v9; // cl
  unsigned __int8 v10; // dl
  int v11; // eax
  unsigned __int8 v12; // cl
  unsigned __int8 v13; // si
  int v14; // eax
  __int64 result; // rax
  unsigned __int8 v16; // cl
  unsigned __int8 v17; // cl
  unsigned __int8 v18; // dl
  int v19; // r12d

  v2 = unk_4F07580;
  *(_BYTE *)(a1 + 14) = 0;
  if ( v2 )
  {
    v4 = *a2;
    *(_BYTE *)(a1 + 12) = *a2;
    v5 = a2[1];
    *(_BYTE *)(a1 + 13) = v5;
    if ( !v5 )
    {
      v6 = a2[2];
      v7 = v4 != 0;
      *(_BYTE *)(a1 + 14) = v6 | 0x80;
      v8 = a2[2];
      v9 = v6 & 0x7F;
      if ( (v8 & 0x7F) != 0 )
        v7 = 1;
      goto LABEL_5;
    }
    v17 = a2[2];
    *(_BYTE *)(a1 + 14) = v17 | 0x80;
    v8 = a2[2];
    v9 = v17 & 0x7F;
    v7 = 1;
    if ( (v8 & 0x7F) == 0 )
    {
LABEL_5:
      a2 += 3;
      goto LABEL_9;
    }
    a2 += 3;
LABEL_17:
    v18 = *a2;
    *(_DWORD *)(a1 + 4) = *a2 >> 7;
    v14 = (2 * (v18 & 0x7F)) | (v8 >> 7);
    if ( v14 != 255 )
    {
      if ( !v14 )
      {
LABEL_19:
        *(_BYTE *)(a1 + 14) = v9;
        v19 = sub_8EE4D0((_BYTE *)(a1 + 12), 24);
        sub_8EE880((_BYTE *)(a1 + 12), 24, v19);
        *(_DWORD *)a1 = 2;
        result = (unsigned int)(-125 - v19);
        *(_DWORD *)(a1 + 28) = 24;
        *(_DWORD *)(a1 + 8) = result;
        return result;
      }
      goto LABEL_20;
    }
    goto LABEL_23;
  }
  v10 = a2[3];
  *(_BYTE *)(a1 + 12) = v10;
  v11 = a2[2];
  *(_BYTE *)(a1 + 13) = v11;
  if ( (_BYTE)v11 )
  {
    v16 = a2[1];
    *(_BYTE *)(a1 + 14) = v16 | 0x80;
    v8 = a2[1];
    v9 = v16 & 0x7F;
    if ( (v8 & 0x7F) == 0 )
      goto LABEL_17;
    v7 = 1;
  }
  else
  {
    v12 = a2[1];
    v7 = v10 != 0;
    *(_BYTE *)(a1 + 14) = v12 | 0x80;
    v8 = a2[1];
    v9 = v12 & 0x7F;
    if ( (v8 & 0x7F) != 0 )
      v7 = 1;
  }
LABEL_9:
  v13 = *a2;
  *(_DWORD *)(a1 + 4) = v13 >> 7;
  v14 = (2 * (v13 & 0x7F)) | (v8 >> 7);
  if ( v14 != 255 )
  {
    if ( !v14 )
    {
      if ( !v7 )
      {
        *(_DWORD *)a1 = 6;
        *(_DWORD *)(a1 + 8) = -126;
        *(_DWORD *)(a1 + 28) = 24;
        return 4294967170LL;
      }
      goto LABEL_19;
    }
LABEL_20:
    result = (unsigned int)(v14 - 126);
    *(_DWORD *)a1 = 2;
    *(_DWORD *)(a1 + 8) = result;
    *(_DWORD *)(a1 + 28) = 24;
    return result;
  }
  if ( !v7 )
  {
    *(_DWORD *)a1 = 4;
    *(_DWORD *)(a1 + 8) = 129;
    *(_DWORD *)(a1 + 28) = 24;
    return 129;
  }
LABEL_23:
  *(_DWORD *)a1 = 3;
  *(_DWORD *)(a1 + 8) = 129;
  *(_DWORD *)(a1 + 28) = 24;
  return 129;
}
