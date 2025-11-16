// Function: sub_74C380
// Address: 0x74c380
//
__int64 __fastcall sub_74C380(__int64 a1, __int64 (__fastcall **a2)(char *, _QWORD))
{
  __int64 v2; // rax

  if ( (*(_BYTE *)(a1 + 124) & 1) == 0 )
  {
    v2 = *(_QWORD *)(a1 + 40);
    if ( v2 )
    {
      if ( *(_BYTE *)(v2 + 28) == 3 )
        sub_74C380(*(_QWORD *)(v2 + 32));
    }
  }
  sub_74C010(a1, 28, (__int64)a2);
  return (*a2)("::", a2);
}
