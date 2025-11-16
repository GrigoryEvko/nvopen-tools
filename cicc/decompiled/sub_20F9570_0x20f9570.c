// Function: sub_20F9570
// Address: 0x20f9570
//
void __fastcall sub_20F9570(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r13

  if ( (*(_BYTE *)(a2 + 229) & 2) == 0 )
  {
    v2 = sub_20F9510(a1, a2);
    v3 = v2;
    if ( v2 )
    {
      if ( (*(_BYTE *)(v2 + 229) & 2) != 0 )
      {
        (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 104LL))(a1, v2);
        (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 88LL))(a1, v3);
      }
    }
  }
}
