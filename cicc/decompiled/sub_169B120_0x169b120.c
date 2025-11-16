// Function: sub_169B120
// Address: 0x169b120
//
__int64 __fastcall sub_169B120(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r13d
  int v4; // r14d
  unsigned int v5; // r14d
  int v6; // r13d
  int v7; // eax
  __int64 result; // rax
  char v9; // al

  if ( *((_DWORD *)a2 + 2) > 0x40u )
    a2 = (__int64 *)*a2;
  v2 = *a2;
  sub_1698320((_QWORD *)a1, (__int64)&unk_42AE9E0);
  v3 = v2;
  v4 = v2;
  LODWORD(v2) = 8 * ((unsigned int)v2 >> 31);
  v5 = v4 & 0x7FFFFF;
  v6 = (unsigned __int8)(v3 >> 23);
  v7 = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  *(_BYTE *)(a1 + 18) = v2 | *(_BYTE *)(a1 + 18) & 0xF7;
  if ( v5 | v6 )
  {
    if ( v5 || v6 != 255 )
    {
      v9 = *(_BYTE *)(a1 + 18) & 0xF8;
      if ( v5 && v6 == 255 )
      {
        *(_BYTE *)(a1 + 18) = v9 | 1;
        result = sub_1698470(a1);
        *(_QWORD *)result = v5;
      }
      else
      {
        *(_BYTE *)(a1 + 18) = v9 | 2;
        *(_WORD *)(a1 + 16) = v6 - 127;
        *(_QWORD *)sub_1698470(a1) = v5;
        if ( v6 )
        {
          result = sub_1698470(a1);
          *(_QWORD *)result |= 0x800000uLL;
        }
        else
        {
          *(_WORD *)(a1 + 16) = -126;
          return 4294967170LL;
        }
      }
    }
    else
    {
      result = v7 & 0xFFFFFFF8;
      *(_BYTE *)(a1 + 18) = result;
    }
  }
  else
  {
    result = v7 & 0xFFFFFFF8 | 3;
    *(_BYTE *)(a1 + 18) = result;
  }
  return result;
}
