// Function: sub_5F8060
// Address: 0x5f8060
//
void __fastcall sub_5F8060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // rax

  if ( !*(_QWORD *)(a1 + 152) )
  {
    for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 183LL) & 0x20) != 0 )
    {
      sub_895AB0(a1, a2, a3, a4, a2);
    }
    else if ( (*(_BYTE *)(a2 + 141) & 0x20) == 0 )
    {
      sub_5E8530((_QWORD *)a2, (*(_DWORD *)(a2 + 176) & 0x11000) == 4096);
    }
  }
}
