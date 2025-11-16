// Function: sub_660E20
// Address: 0x660e20
//
__int64 __fastcall sub_660E20(__int16 a1, int a2, int a3, int a4, __int64 a5, __int64 a6, _QWORD *a7)
{
  _QWORD *v7; // r12
  __int64 result; // rax
  _QWORD v9[62]; // [rsp+0h] [rbp-1F0h] BYREF

  v7 = a7;
  if ( !a7 )
  {
    v7 = v9;
    memset(v9, 0, 0x1D8u);
    v9[19] = v9;
    v9[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC )
    {
      if ( qword_4F077A8 <= 0x9F5Fu )
        BYTE2(v9[22]) |= 1u;
    }
  }
  *((_DWORD *)v7 + 31) = *((_DWORD *)v7 + 31) & 0xFFFABFFF
                       | (a4 << 18) & 0x40000
                       | (a1 << 14) & 0x4000
                       | (a3 << 16) & 0x10000;
  if ( a2 )
  {
    *((_BYTE *)v7 + 125) |= 0x80u;
    v7[46] = a5;
  }
  if ( a3 && dword_4F04C34 == dword_4F04C64 )
    *((_BYTE *)v7 + 132) |= 0x80u;
  result = sub_662DE0(v7, a6);
  if ( *v7 )
  {
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(result + 14) & 0x10) != 0 )
    {
      result = *(_BYTE *)(sub_87D520(*v7) + 88) & 0x70;
      if ( (_BYTE)result == 16 )
        return sub_6851C0(3109, v7 + 6);
    }
  }
  return result;
}
