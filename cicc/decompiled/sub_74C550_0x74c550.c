// Function: sub_74C550
// Address: 0x74c550
//
void __fastcall sub_74C550(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  void (__fastcall *v3)(__int64, _QWORD); // rax
  __int64 v4; // [rsp+8h] [rbp-18h]

  v3 = *(void (__fastcall **)(__int64, _QWORD))(a3 + 24);
  if ( v3 )
  {
    v3(a1, a2);
  }
  else
  {
    if ( !dword_4F072C8
      && (a2 != 6
       || (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u
       || (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 109LL) & 0x20) == 0) )
    {
      v4 = a3;
      sub_74C480(*(_QWORD *)(a1 + 40), a3);
      a3 = v4;
    }
    sub_74C010(a1, a2, a3);
  }
}
