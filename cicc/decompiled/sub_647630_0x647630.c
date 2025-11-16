// Function: sub_647630
// Address: 0x647630
//
__int64 __fastcall sub_647630(unsigned __int8 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // [rsp+8h] [rbp-18h]
  unsigned int v6; // [rsp+Ch] [rbp-14h]

  if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)a3 + 4) == 1
    && a1 != 7
    && dword_4F077C4 != 2
    && a1 != 2
    && (*(_BYTE *)(a2 + 17) & 0x20) == 0 )
  {
    v5 = a4;
    v6 = a3;
    sub_684B30(231, a2 + 8);
    a4 = v5;
    a3 = v6;
  }
  return sub_885AD0(a1, a2, a3, a4);
}
