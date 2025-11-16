// Function: sub_6E45A0
// Address: 0x6e45a0
//
void __fastcall sub_6E45A0(__int64 a1)
{
  char v1; // al
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax

  if ( *(_QWORD *)(a1 + 128) )
  {
    v1 = *(_BYTE *)(a1 + 16);
    if ( v1 == 1 )
    {
      v2 = *(_QWORD *)(a1 + 144);
      if ( v2 )
        *(_BYTE *)(v2 + 26) |= 4u;
    }
    else if ( v1 == 2 )
    {
      v3 = *(_QWORD *)(a1 + 288);
      if ( v3 )
      {
        *(_BYTE *)(v3 + 26) |= 4u;
      }
      else if ( *(_BYTE *)(a1 + 317) == 12 && *(_BYTE *)(a1 + 320) == 1 )
      {
        v4 = sub_72E9A0(a1 + 144);
        if ( v4 )
          *(_BYTE *)(v4 + 26) |= 4u;
      }
    }
  }
}
