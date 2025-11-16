// Function: sub_169AE50
// Address: 0x169ae50
//
__int64 __fastcall sub_169AE50(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // r14
  __int64 v3; // r12
  unsigned __int64 v4; // r15
  __int64 v5; // rbx
  int v6; // eax
  __int64 v7; // r15
  unsigned int v8; // eax
  __int64 result; // rax

  if ( *((_DWORD *)a2 + 2) > 0x40u )
    a2 = (__int64 *)*a2;
  v2 = a2[1];
  v3 = *a2;
  sub_1698320((_QWORD *)a1, (__int64)&unk_42AE9C0);
  v4 = v2;
  v5 = v2 & 0xFFFFFFFFFFFFLL;
  LODWORD(v2) = 8 * (v2 >> 63);
  v6 = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  v7 = HIWORD(v4) & 0x7FFF;
  *(_BYTE *)(a1 + 18) = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  if ( v7 )
  {
    if ( v7 == 0x7FFF )
    {
      result = v6 & 0xFFFFFFF8;
      if ( v5 | v3 )
      {
        *(_BYTE *)(a1 + 18) = result | 1;
        *(_QWORD *)sub_1698470(a1) = v3;
        result = sub_1698470(a1);
        *(_QWORD *)(result + 8) = v5;
      }
      else
      {
        *(_BYTE *)(a1 + 18) = result;
      }
    }
    else
    {
      *(_WORD *)(a1 + 16) = v7 - 0x3FFF;
      *(_BYTE *)(a1 + 18) = v6 & 0xF8 | 2;
      *(_QWORD *)sub_1698470(a1) = v3;
      *(_QWORD *)(sub_1698470(a1) + 8) = v5;
      result = sub_1698470(a1);
      *(_QWORD *)(result + 8) |= 0x1000000000000uLL;
    }
  }
  else
  {
    v8 = v6 & 0xFFFFFFF8;
    if ( v5 | v3 )
    {
      *(_BYTE *)(a1 + 18) = v8 | 2;
      *(_WORD *)(a1 + 16) = -16383;
      *(_QWORD *)sub_1698470(a1) = v3;
      result = sub_1698470(a1);
      *(_QWORD *)(result + 8) = v5;
      *(_WORD *)(a1 + 16) = -16382;
    }
    else
    {
      result = v8 | 3;
      *(_BYTE *)(a1 + 18) = result;
    }
  }
  return result;
}
