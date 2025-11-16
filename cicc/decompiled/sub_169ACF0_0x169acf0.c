// Function: sub_169ACF0
// Address: 0x169acf0
//
__int64 __fastcall sub_169ACF0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r14
  int v5; // eax
  __int64 result; // rax

  if ( *(_DWORD *)(a2 + 8) > 0x40u )
    a2 = *(_QWORD *)a2;
  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(_QWORD *)a2;
  sub_1698320((_QWORD *)a1, (__int64)&unk_42AE9B0);
  v4 = v2 & 0x7FFF;
  LODWORD(v2) = 8 * ((v2 >> 15) & 1);
  v5 = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  *(_BYTE *)(a1 + 18) = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  if ( v4 | v3 )
  {
    if ( v3 == 0x8000000000000000LL && v4 == 0x7FFF )
    {
      result = v5 & 0xFFFFFFF8;
      *(_BYTE *)(a1 + 18) = result;
    }
    else if ( v3 != 0x8000000000000000LL && v4 == 0x7FFF || v4 != 0 && v4 != 0x7FFF && v3 >= 0 )
    {
      *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 1;
      *(_QWORD *)sub_1698470(a1) = v3;
      result = sub_1698470(a1);
      *(_QWORD *)(result + 8) = 0;
    }
    else
    {
      *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 2;
      *(_WORD *)(a1 + 16) = v4 - 0x3FFF;
      *(_QWORD *)sub_1698470(a1) = v3;
      result = sub_1698470(a1);
      *(_QWORD *)(result + 8) = 0;
      if ( !v4 )
      {
        *(_WORD *)(a1 + 16) = -16382;
        return 4294950914LL;
      }
    }
  }
  else
  {
    result = v5 & 0xFFFFFFF8 | 3;
    *(_BYTE *)(a1 + 18) = result;
  }
  return result;
}
