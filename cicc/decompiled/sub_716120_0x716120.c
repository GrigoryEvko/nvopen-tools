// Function: sub_716120
// Address: 0x716120
//
__int64 __fastcall sub_716120(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  unsigned int v3; // r12d
  _QWORD v5[4]; // [rsp+10h] [rbp-20h] BYREF

  if ( (*(_BYTE *)(a1 + 25) & 3) != 0 )
    return 0;
  v2 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  if ( (dword_4F04C44 != -1 || (v2[6] & 6) != 0 || v2[4] == 12) && (v2[12] & 0x10) == 0 && (unsigned int)sub_731EE0(a1) )
    return 0;
  v5[0] = 0;
  v5[1] = 0;
  v3 = sub_7A30C0(a1, 0, 0, a2);
  sub_67E3D0(v5);
  return v3;
}
