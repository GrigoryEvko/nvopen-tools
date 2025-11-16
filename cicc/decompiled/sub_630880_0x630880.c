// Function: sub_630880
// Address: 0x630880
//
void __fastcall sub_630880(__int64 *a1, __int64 a2)
{
  unsigned __int8 v3; // di
  __int64 v4; // rax
  __int64 v5; // rdx

  if ( !*a1 )
    *a1 = sub_72C9A0();
  if ( a2 )
    *((_BYTE *)a1 + 41) |= 4u;
  v3 = 2;
  if ( (*((_BYTE *)a1 + 41) & 4) != 0 )
    v3 = 4 * (*(_BYTE *)(*a1 + 173) == 10) + 2;
  if ( (a1[5] & 0x408) != 0 )
  {
    if ( unk_4D03C50 )
      v4 = sub_6EAFA0(v3);
    else
      v4 = sub_725A70(v3);
    v5 = *a1;
    a1[1] = v4;
    *(_QWORD *)(v4 + 56) = v5;
    *(_BYTE *)(a1[1] + 50) = ~*(_BYTE *)(*a1 + 169) & 0x40 | *(_BYTE *)(a1[1] + 50) & 0xBF;
    *(_BYTE *)(a1[1] + 50) = (*((_BYTE *)a1 + 41) >> 4 << 7) | *(_BYTE *)(a1[1] + 50) & 0x7F;
    if ( a2 )
    {
      *(_QWORD *)(a1[1] + 16) = a2;
      if ( (*((_BYTE *)a1 + 42) & 0x20) == 0 )
        *(_BYTE *)(a2 + 193) |= 0x40u;
    }
    *a1 = 0;
  }
}
