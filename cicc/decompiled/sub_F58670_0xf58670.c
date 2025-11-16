// Function: sub_F58670
// Address: 0xf58670
//
void __fastcall sub_F58670(__int64 a1, __int64 *a2)
{
  __int64 v2; // r12
  char *v3; // rax
  size_t v4; // rdx
  __int64 *v5; // rax
  unsigned int v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = *(_QWORD *)(a1 - 32);
  if ( v2 )
  {
    if ( !*(_BYTE *)v2
      && *(_QWORD *)(a1 + 80) == *(_QWORD *)(v2 + 24)
      && (*(_BYTE *)(v2 + 32) & 0xFu) - 7 > 1
      && (*(_BYTE *)(v2 + 7) & 0x10) != 0 )
    {
      v3 = (char *)sub_BD5D20(*(_QWORD *)(a1 - 32));
      if ( (unsigned __int8)sub_980AF0(*a2, v3, v4, v6) )
      {
        if ( (unsigned __int8)sub_F50940(a2, v6[0]) )
        {
          if ( !sub_B2DCC0(v2) )
          {
            v5 = (__int64 *)sub_BD5C60(a1);
            *(_QWORD *)(a1 + 72) = sub_A7A090((__int64 *)(a1 + 72), v5, -1, 23);
          }
        }
      }
    }
  }
}
