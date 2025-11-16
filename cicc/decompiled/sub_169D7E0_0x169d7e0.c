// Function: sub_169D7E0
// Address: 0x169d7e0
//
__int64 __fastcall sub_169D7E0(__int64 a1, __int64 *a2)
{
  __int16 *v2; // rax

  v2 = (__int16 *)*a2;
  if ( (_UNKNOWN *)*a2 == &unk_42AE9F0 )
  {
    sub_169AC10(a1, (__int64)a2);
    return a1;
  }
  else if ( v2 == (__int16 *)&unk_42AE9E0 )
  {
    sub_169AB30(a1, (__int64)a2);
    return a1;
  }
  else if ( v2 == word_42AE9D0 )
  {
    sub_169AA10(a1, (__int64)a2);
    return a1;
  }
  else
  {
    if ( v2 == (__int16 *)&unk_42AE9C0 )
    {
      sub_169A8B0(a1, (__int64)a2);
      return a1;
    }
    if ( v2 != word_42AE980 )
    {
      sub_169A7C0(a1, (__int64)a2);
      return a1;
    }
    sub_169D620(a1, a2);
    return a1;
  }
}
