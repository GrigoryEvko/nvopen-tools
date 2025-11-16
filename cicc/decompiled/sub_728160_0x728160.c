// Function: sub_728160
// Address: 0x728160
//
void __fastcall sub_728160(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // al

  if ( *(_QWORD *)(a1 + 16) )
    goto LABEL_2;
  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    if ( (*(_BYTE *)(v2 + 173) & 4) != 0 )
      goto LABEL_2;
  }
  v3 = *(_BYTE *)(a1 + 48);
  if ( v3 == 2 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 56) + 173LL) )
      return;
LABEL_2:
    *(_DWORD *)(a2 + 80) = 1;
    *(_DWORD *)(a2 + 72) = 1;
    return;
  }
  if ( v3 == 5 )
    goto LABEL_2;
}
