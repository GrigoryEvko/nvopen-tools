// Function: sub_128B370
// Address: 0x128b370
//
__int64 __fastcall sub_128B370(__int64 *a1, _QWORD *a2, unsigned __int8 a3, unsigned __int64 a4, _DWORD *a5)
{
  __int64 v7; // rax
  char v8; // bl
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int8 v11; // al
  __int64 v13; // [rsp+0h] [rbp-40h]

  v7 = sub_127A030(*(_QWORD *)(*a1 + 32) + 8LL, a4, 0);
  v8 = *(_BYTE *)(a4 + 140);
  v9 = v7;
  if ( v8 == 12 )
  {
    v10 = a4;
    do
    {
      v10 = *(_QWORD *)(v10 + 160);
      v8 = *(_BYTE *)(v10 + 140);
    }
    while ( v8 == 12 );
  }
  v13 = v9;
  v11 = sub_127B3A0(a4);
  return sub_128A450(a1, a2, a3, v13, v11, v8 == 1, a5);
}
