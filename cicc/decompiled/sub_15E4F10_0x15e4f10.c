// Function: sub_15E4F10
// Address: 0x15e4f10
//
__int64 __fastcall sub_15E4F10(__int64 a1)
{
  char v1; // al
  __int64 v2; // rax
  char v3; // dl

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 1 )
  {
    v2 = sub_164A820(*(_QWORD *)(a1 - 24));
    v3 = *(_BYTE *)(v2 + 16);
    if ( v3 != 3 && v3 )
      return 0;
    else
      return *(_QWORD *)(v2 + 48);
  }
  else if ( v1 == 2 )
  {
    return 0;
  }
  else
  {
    return *(_QWORD *)(a1 + 48);
  }
}
