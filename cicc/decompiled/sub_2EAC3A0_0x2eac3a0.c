// Function: sub_2EAC3A0
// Address: 0x2eac3a0
//
__int64 __fastcall sub_2EAC3A0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax

  v2 = sub_2E79000(a2);
  *(_QWORD *)a1 = 0;
  LODWORD(v2) = *(_DWORD *)(v2 + 4);
  *(_BYTE *)(a1 + 20) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = v2;
  return a1;
}
