// Function: sub_7279A0
// Address: 0x7279a0
//
__int64 __fastcall sub_7279A0(__int64 a1)
{
  int v1; // r13d
  __int64 v2; // rax
  __int64 v3; // r12
  int v4; // ecx

  v1 = unk_4D03FE8;
  unk_4D03FE8 = 1;
  if ( !v1 )
    sub_727950();
  v2 = dword_4F07988 + sub_822F50(1, a1 + dword_4F0798C);
  if ( !unk_4D03FE8 )
  {
    *(_QWORD *)v2 = 0;
    v2 += 8;
  }
  *(_QWORD *)v2 = 0;
  v3 = v2 + 16;
  v4 = unk_4D03FE8;
  unk_4D03FE8 = v1;
  *(_BYTE *)(v2 + 8) = (2 * (v4 == 0)) | (8 * (unk_4F06CFC & 1)) | 1;
  if ( v1 )
    return v2 + 16;
  sub_727950();
  return v3;
}
