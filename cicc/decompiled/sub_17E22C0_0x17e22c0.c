// Function: sub_17E22C0
// Address: 0x17e22c0
//
__int64 *__fastcall sub_17E22C0(__int64 *a1, int a2)
{
  __int64 v2; // rax

  v2 = sub_22077B0(104);
  if ( v2 )
  {
    *(_QWORD *)v2 = v2;
    *(_QWORD *)(v2 + 40) = v2 + 56;
    *(_DWORD *)(v2 + 8) = a2;
    *(_DWORD *)(v2 + 12) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_BYTE *)(v2 + 24) = 0;
    *(_QWORD *)(v2 + 28) = 0;
    *(_QWORD *)(v2 + 48) = 0x200000000LL;
    *(_QWORD *)(v2 + 72) = v2 + 88;
    *(_QWORD *)(v2 + 80) = 0x200000000LL;
  }
  *a1 = v2;
  return a1;
}
