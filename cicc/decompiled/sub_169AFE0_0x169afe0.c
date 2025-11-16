// Function: sub_169AFE0
// Address: 0x169afe0
//
__int64 __fastcall sub_169AFE0(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // r14
  int v6; // eax
  __int64 result; // rax

  if ( *((_DWORD *)a2 + 2) > 0x40u )
    a2 = (__int64 *)*a2;
  v2 = *a2;
  sub_1698320((_QWORD *)a1, (__int64)&unk_42AE9D0);
  v3 = v2;
  v4 = v2 & 0xFFFFFFFFFFFFFLL;
  LODWORD(v2) = 8 * (v2 >> 63);
  v5 = (v3 >> 52) & 0x7FF;
  v6 = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  *(_BYTE *)(a1 + 18) = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  if ( !(v4 | v5) )
  {
    result = v6 & 0xFFFFFFF8 | 3;
    *(_BYTE *)(a1 + 18) = result;
    return result;
  }
  if ( v4 )
  {
    if ( v5 == 2047 )
    {
      *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 1;
      result = sub_1698470(a1);
      *(_QWORD *)result = v4;
      return result;
    }
  }
  else if ( v5 == 2047 )
  {
    result = v6 & 0xFFFFFFF8;
    *(_BYTE *)(a1 + 18) = result;
    return result;
  }
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 2;
  *(_WORD *)(a1 + 16) = v5 - 1023;
  *(_QWORD *)sub_1698470(a1) = v4;
  if ( v5 )
  {
    result = sub_1698470(a1);
    *(_QWORD *)result |= 0x10000000000000uLL;
  }
  else
  {
    *(_WORD *)(a1 + 16) = -1022;
    return 4294966274LL;
  }
  return result;
}
