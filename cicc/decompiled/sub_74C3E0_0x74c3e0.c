// Function: sub_74C3E0
// Address: 0x74c3e0
//
__int64 __fastcall sub_74C3E0(__int64 a1, __int64 (__fastcall **a2)(char *, _QWORD))
{
  __int64 result; // rax
  __int64 v4; // rdi

  sub_74C480(*(_QWORD *)(a1 + 40));
  result = *(_QWORD *)(a1 + 168);
  if ( !*(_BYTE *)(result + 113) )
  {
    v4 = *(_QWORD *)(result + 256);
    if ( !v4 )
      v4 = a1;
    sub_74C010(v4, 6, (__int64)a2);
    return (*a2)("::", a2);
  }
  return result;
}
