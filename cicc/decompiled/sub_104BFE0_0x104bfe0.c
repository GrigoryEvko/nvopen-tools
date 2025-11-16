// Function: sub_104BFE0
// Address: 0x104bfe0
//
__int64 __fastcall sub_104BFE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi

  v2 = a1 + 176;
  *(_QWORD *)(v2 + 128) = a2;
  *(_DWORD *)(v2 + 144) = *(_DWORD *)(a2 + 92);
  sub_B29120(v2);
  return 0;
}
