// Function: sub_5E6640
// Address: 0x5e6640
//
_BOOL8 __fastcall sub_5E6640(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // r12
  int v5; // r14d
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // r15d
  int v14; // eax

  v3 = a1;
  *a3 = 0;
  v5 = sub_8D3070();
  if ( v5 )
  {
    v10 = sub_73C570(a2, 1, -1);
    v9 = sub_72D600(v10);
  }
  else
  {
    if ( !unk_4D0446C )
    {
      if ( !dword_4F077BC )
        return 0;
      if ( dword_4F077B4 )
      {
        if ( unk_4F077A0 <= 0x752Fu )
          return 0;
      }
      else if ( qword_4F077A8 <= 0x9E33u )
      {
        return 0;
      }
    }
    if ( !(unsigned int)sub_8D3110(a1) )
      return 0;
    v9 = sub_72D6A0(a2);
  }
  if ( !v9 )
    return 0;
  if ( v9 == a1 || (unsigned int)sub_8D97D0(a1, v9, 0, v7, v8) )
    return 1;
  if ( (unsigned int)sub_8D2FB0(a1) && (unsigned int)sub_8D2FB0(v9) )
  {
    v3 = sub_8D46C0(a1);
    v9 = sub_8D46C0(v9);
    if ( (*(_BYTE *)(v3 + 140) & 0xFB) == 8 )
    {
      v13 = sub_8D4C10(v3, unk_4F077C4 != 2);
      if ( (v13 & 0xFFFFFFFC) != 0 )
        return 0;
    }
    else
    {
      v13 = 0;
    }
    if ( v3 == v9 )
      goto LABEL_19;
  }
  else if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
  {
    v13 = sub_8D4C10(a1, unk_4F077C4 != 2);
    if ( (v13 & 0xFFFFFFFC) != 0 )
      return 0;
  }
  else
  {
    v13 = 0;
  }
  if ( !(unsigned int)sub_8D97D0(v3, v9, 32, v11, v12) )
    return 0;
  v3 = v9;
LABEL_19:
  if ( (*(_BYTE *)(v3 + 140) & 0xFB) == 8 )
  {
    v14 = sub_8D4C10(v3, unk_4F077C4 != 2);
    if ( (v14 & 1) != 0 && (v13 ^ v14) == 1 && v5 )
      return 1;
  }
  *a3 = 1;
  if ( unk_4F077C4 != 2 )
    return 0;
  return unk_4F07778 > 202001;
}
