// Function: sub_38708D0
// Address: 0x38708d0
//
char __fastcall sub_38708D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax

  while ( 1 )
  {
    LOBYTE(v8) = sub_15CCEE0(a2, a3, a4);
    if ( (_BYTE)v8 )
      break;
    sub_38707D0(a1, a3);
    sub_15F22F0((_QWORD *)a3, a4);
    if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
    {
      a4 = a3;
      v8 = **(_QWORD **)(a3 - 8);
      if ( a5 == v8 )
        return v8;
    }
    else
    {
      a4 = a3;
      v8 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
      if ( a5 == v8 )
        return v8;
    }
    a3 = v8;
  }
  return v8;
}
