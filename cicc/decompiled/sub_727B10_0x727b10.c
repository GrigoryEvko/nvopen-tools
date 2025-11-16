// Function: sub_727B10
// Address: 0x727b10
//
__int64 __fastcall sub_727B10(__int64 a1, __int64 a2)
{
  int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // r12
  int v5; // ecx

  v2 = unk_4D03FE8;
  unk_4D03FE8 = 0;
  if ( v2 )
    sub_727950();
  v3 = dword_4F07988 + sub_822F50(*(unsigned int *)(a2 + 400), a1 + dword_4F0798C);
  if ( !unk_4D03FE8 )
  {
    *(_QWORD *)v3 = 0;
    v3 += 8;
  }
  *(_QWORD *)v3 = 0;
  v4 = v3 + 16;
  v5 = unk_4D03FE8;
  unk_4D03FE8 = v2;
  *(_BYTE *)(v3 + 8) = (2 * (v5 == 0)) | (8 * (unk_4F06CFC & 1)) | 1;
  if ( !v2 )
    return v3 + 16;
  sub_727950();
  return v4;
}
