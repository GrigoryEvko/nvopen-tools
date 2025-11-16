// Function: sub_773640
// Address: 0x773640
//
void __fastcall sub_773640(__int64 a1)
{
  _QWORD *v1; // rax
  int v2; // edx

  v1 = *(_QWORD **)(a1 + 184);
  if ( v1 )
  {
    v2 = 0;
    do
    {
      v1 = (_QWORD *)*v1;
      ++v2;
    }
    while ( v1 );
    if ( v2 && (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_67E440(0xBB3u, (_DWORD *)(a1 + 112), v2, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0xBB1u, (FILE *)(*(_QWORD *)(a1 + 184) + 24LL), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
    }
  }
}
