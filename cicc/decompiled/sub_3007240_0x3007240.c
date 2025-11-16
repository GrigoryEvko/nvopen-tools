// Function: sub_3007240
// Address: 0x3007240
//
__int64 __fastcall sub_3007240(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // [rsp+0h] [rbp-8h]

  v1 = *(_QWORD *)(a1 + 8);
  BYTE4(v3) = *(_BYTE *)(v1 + 8) == 18;
  LODWORD(v3) = *(_DWORD *)(v1 + 32);
  return v3;
}
