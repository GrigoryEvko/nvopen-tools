// Function: sub_E6E2A0
// Address: 0xe6e2a0
//
unsigned __int64 __fastcall sub_E6E2A0(_QWORD *a1, unsigned __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 result; // rax
  unsigned int v5; // r10d
  _BYTE *v6; // r12
  size_t v7; // r11
  unsigned int v8; // r10d
  __int64 *v9; // r8
  __int64 v10; // r9
  __int64 v11; // r8

  if ( a3 || (result = a2, a4 != -1) )
  {
    v5 = *(_DWORD *)(a2 + 148);
    v6 = *(_BYTE **)(a2 + 128);
    v7 = *(_QWORD *)(a2 + 136);
    if ( a3 )
    {
      v8 = v5 | 0x1000;
      if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
      {
        v9 = *(__int64 **)(a3 - 8);
        v10 = *v9;
        v11 = (__int64)(v9 + 3);
      }
      else
      {
        v10 = 0;
        v11 = 0;
      }
      return sub_E6DEB0(a1, v6, v7, v8, v11, v10, 5u, a4);
    }
    else
    {
      return sub_E6DEB0(a1, v6, v7, v5, (__int64)byte_3F871B3, 0, 0, a4);
    }
  }
  return result;
}
