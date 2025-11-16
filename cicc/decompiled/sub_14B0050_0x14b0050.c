// Function: sub_14B0050
// Address: 0x14b0050
//
__int64 __fastcall sub_14B0050(_QWORD **a1, _BYTE *a2)
{
  unsigned int v2; // r12d
  unsigned __int8 v3; // al
  __int64 v5; // rax

  v3 = a2[16];
  if ( v3 == 14 )
  {
    v2 = 1;
    **a1 = a2 + 24;
    return v2;
  }
  LOBYTE(v2) = v3 <= 0x10u && *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16;
  if ( (_BYTE)v2 )
  {
    v5 = sub_15A1020(a2);
    if ( v5 )
    {
      if ( *(_BYTE *)(v5 + 16) == 14 )
      {
        **a1 = v5 + 24;
        return v2;
      }
    }
  }
  return 0;
}
