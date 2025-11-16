// Function: sub_8D7480
// Address: 0x8d7480
//
_BOOL8 __fastcall sub_8D7480(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  char v4; // al
  char v5; // dl
  unsigned int v6; // r13d
  int v7; // eax
  _BOOL8 result; // rax
  int v9; // eax
  char v10; // al
  char v11; // al
  char v12; // dl

  v2 = a2;
  v3 = a1;
  v4 = *(_BYTE *)(a1 + 140);
  if ( v4 != 2 )
  {
    if ( (v4 & 0xFB) != 8 )
    {
      v5 = *(_BYTE *)(a2 + 140);
      v6 = 0;
      if ( (v5 & 0xFB) == 8 )
        goto LABEL_4;
LABEL_10:
      result = v6 == 0;
      if ( *(_BYTE *)(a1 + 140) != 12 )
        goto LABEL_7;
      do
LABEL_11:
        v3 = *(_QWORD *)(v3 + 160);
      while ( *(_BYTE *)(v3 + 140) == 12 );
LABEL_5:
      if ( v5 == 12 )
      {
        do
          v2 = *(_QWORD *)(v2 + 160);
        while ( *(_BYTE *)(v2 + 140) == 12 );
      }
LABEL_7:
      if ( !result )
        return result;
      goto LABEL_15;
    }
    v9 = sub_8D4C10(a1, dword_4F077C4 != 2);
    v5 = *(_BYTE *)(a2 + 140);
    v6 = v9 & 0xFFFFFF8F;
    if ( (v5 & 0xFB) != 8 )
      goto LABEL_10;
LABEL_4:
    v7 = sub_8D4C10(a2, dword_4F077C4 != 2);
    v5 = *(_BYTE *)(a2 + 140);
    result = (v7 & 0xFFFFFF8F) == v6;
    if ( *(_BYTE *)(a1 + 140) != 12 )
      goto LABEL_5;
    goto LABEL_11;
  }
  v10 = *(_BYTE *)(a2 + 140);
  if ( v10 != 2 )
  {
    v6 = 0;
    if ( (v10 & 0xFB) == 8 )
      goto LABEL_4;
  }
LABEL_15:
  v11 = *(_BYTE *)(v3 + 160);
  if ( (unsigned __int8)(v11 - 1) <= 1u )
  {
    v11 = 0;
  }
  else
  {
    switch ( v11 )
    {
      case 4:
        v11 = 3;
        break;
      case 6:
        v11 = 5;
        break;
      case 8:
        v11 = 7;
        break;
      case 10:
        v11 = 9;
        break;
    }
  }
  v12 = *(_BYTE *)(v2 + 160);
  if ( (unsigned __int8)(v12 - 1) <= 1u )
  {
    v12 = 0;
  }
  else
  {
    switch ( v12 )
    {
      case 4:
        v12 = 3;
        break;
      case 6:
        v12 = 5;
        break;
      case 8:
        v12 = 7;
        break;
      case 10:
        v12 = 9;
        break;
    }
  }
  return v12 == v11;
}
