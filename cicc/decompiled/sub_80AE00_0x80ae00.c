// Function: sub_80AE00
// Address: 0x80ae00
//
void __fastcall sub_80AE00(__int64 a1, char a2)
{
  __int64 *i; // rbx
  __int64 v3; // rax

  if ( *(_BYTE *)(a1 + 89) >> 7 != dword_4F18B88 )
  {
    *(_BYTE *)(a1 + 89) = ((_BYTE)dword_4F18B88 << 7) | *(_BYTE *)(a1 + 89) & 0x7F;
    if ( qword_4F18B90 != a1 )
    {
      if ( a2 == 11 )
      {
        if ( (*(_BYTE *)(a1 + 195) & 1) == 0 )
        {
          if ( (*(_BYTE *)(a1 + 201) & 0x10) == 0 )
            sub_80AD10(a1);
          v3 = *(_QWORD *)(a1 + 104);
          if ( v3 && (*(_BYTE *)(v3 + 11) & 0x20) != 0 )
            sub_80A9A0(*(_QWORD **)(v3 + 32));
        }
      }
      else if ( a2 == 6 )
      {
        sub_8D9600(a1, sub_809E90, 27);
      }
      else if ( a2 == 28 && (*(_BYTE *)(a1 + 124) & 2) != 0 )
      {
        for ( i = *(__int64 **)(a1 + 104); i; i = (__int64 *)*i )
        {
          if ( *((_BYTE *)i + 8) == 80 )
            sub_80A9A0((_QWORD *)i[4]);
        }
      }
    }
  }
}
