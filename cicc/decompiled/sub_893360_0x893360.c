// Function: sub_893360
// Address: 0x893360
//
__int64 sub_893360()
{
  __int64 v0; // rax
  __int64 v1; // rcx
  char v2; // al
  __int64 v3; // rdx

  if ( dword_4F04C64 < 0 )
    return *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64 + 4;
  while ( 1 )
  {
    if ( *(_BYTE *)v0 == 9 )
    {
      v1 = *(_QWORD *)(v0 + 356);
      if ( v1 )
        break;
    }
    v0 -= 776;
    if ( qword_4F04C68[0] - 772LL == v0 )
      return *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
  }
  v2 = *(_BYTE *)(v1 + 80);
  if ( v2 == 6 || v2 == 3 )
    return *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
  v3 = *(_QWORD *)(v1 + 96);
  if ( (unsigned __int8)(v2 - 4) <= 1u )
    return *(_QWORD *)(v3 + 120);
  else
    return *(_QWORD *)(v3 + 48);
}
