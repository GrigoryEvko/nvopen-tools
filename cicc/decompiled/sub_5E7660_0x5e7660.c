// Function: sub_5E7660
// Address: 0x5e7660
//
_BOOL8 __fastcall sub_5E7660(__int64 a1)
{
  __int64 v1; // rbx
  int v2; // r8d
  _BOOL8 result; // rax
  char v4; // al

  if ( !a1 )
    return 0;
  v1 = a1;
  v2 = sub_884000(a1, 1);
  result = 1;
  if ( v2 )
  {
    v4 = *(_BYTE *)(a1 + 80);
    if ( v4 == 16 )
    {
      v1 = **(_QWORD **)(a1 + 88);
      v4 = *(_BYTE *)(v1 + 80);
    }
    if ( v4 == 24 )
    {
      v1 = *(_QWORD *)(v1 + 88);
      v4 = *(_BYTE *)(v1 + 80);
    }
    return v4 == 10 && (*(_BYTE *)(*(_QWORD *)(v1 + 88) + 206LL) & 0x10) != 0;
  }
  return result;
}
