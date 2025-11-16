// Function: sub_9C3590
// Address: 0x9c3590
//
__int64 *__fastcall sub_9C3590(__int64 *a1, int a2)
{
  __int64 v2; // rax

  v2 = sub_22077B0(72);
  if ( v2 )
  {
    *(_DWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 40) = v2 + 56;
    *(_DWORD *)(v2 + 12) = a2;
    *(_QWORD *)(v2 + 16) = 0;
    *(_QWORD *)(v2 + 24) = 0;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)v2 = &unk_49D9790;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 56) = 0;
    *(_QWORD *)(v2 + 64) = 0;
  }
  *a1 = v2;
  return a1;
}
