// Function: sub_6E0040
// Address: 0x6e0040
//
__int64 __fastcall sub_6E0040(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  unsigned __int8 v4; // dl
  char v5; // cl
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  _BOOL4 v10; // esi
  __int64 v11; // rdx
  __int64 v12; // rdx

  result = *(unsigned __int8 *)(a1 + 24);
  if ( (_BYTE)result != 1 )
  {
    if ( (_BYTE)result != 17 )
      return result;
    goto LABEL_3;
  }
  v4 = *(_BYTE *)(a1 + 56);
  v5 = *(_BYTE *)(a1 + 60);
  v7 = *(_QWORD *)(a1 + 72);
  if ( v4 > 0x5Bu )
  {
    if ( (unsigned __int8)(v4 - 103) <= 1u )
      goto LABEL_23;
  }
  else if ( v4 > 0x56u )
  {
    goto LABEL_23;
  }
  if ( (v5 & 3) == 0 )
  {
    v10 = 0;
    goto LABEL_24;
  }
  if ( (unsigned __int8)(v4 - 105) > 4u )
  {
LABEL_23:
    v10 = 1;
LABEL_24:
    if ( (*(_BYTE *)(a1 + 60) & 2) == 0 )
    {
      result = sub_6DFF60(*(_QWORD *)(a1 + 72), v10, a2);
      if ( (*(_BYTE *)(a1 + 60) & 2) == 0 || !v7 )
        goto LABEL_3;
      goto LABEL_18;
    }
    if ( !v7 )
    {
      result = sub_6DFF60(0, v10, a2);
      goto LABEL_3;
    }
    v8 = *(_QWORD *)(v7 + 16);
    goto LABEL_14;
  }
  v8 = *(_QWORD *)(v7 + 16);
  v9 = v8;
  if ( (unsigned __int8)(v4 - 106) <= 3u )
    v9 = *(_QWORD *)(v8 + 16);
  v10 = 1;
  if ( v9 )
    v10 = *(_QWORD *)(v9 + 16) == 0;
  if ( (v5 & 2) != 0 )
  {
LABEL_14:
    v11 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v7 + 16) = v11;
      if ( !v8 )
        break;
      v11 = v7;
      v7 = v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
  }
  result = sub_6DFF60(v7, v10, a2);
  if ( (*(_BYTE *)(a1 + 60) & 2) != 0 )
  {
LABEL_18:
    v12 = 0;
    while ( 1 )
    {
      result = *(_QWORD *)(v7 + 16);
      *(_QWORD *)(v7 + 16) = v12;
      v12 = v7;
      if ( !result )
        break;
      v7 = result;
    }
  }
LABEL_3:
  a2[19] = 1;
  return result;
}
