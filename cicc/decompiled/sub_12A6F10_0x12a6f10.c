// Function: sub_12A6F10
// Address: 0x12a6f10
//
__int64 __fastcall sub_12A6F10(__int64 a1, unsigned int a2, char *a3, char *a4, _DWORD *a5)
{
  __int64 v7; // rbx
  unsigned int v8; // ebx
  _DWORD v10[13]; // [rsp+Ch] [rbp-34h] BYREF

  if ( *(_BYTE *)(a1 + 24) != 2 )
    sub_127B550(a3, a5, 1);
  v7 = *(_QWORD *)(a1 + 56);
  if ( *(_BYTE *)(v7 + 173) != 1 )
    sub_127B550(a3, a5, 1);
  v8 = sub_620FA0(v7, v10);
  if ( a2 < v8 || v10[0] )
    sub_127B550(a4, a5, 1);
  return v8;
}
