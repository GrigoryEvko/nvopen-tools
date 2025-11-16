// Function: sub_2352120
// Address: 0x2352120
//
__int64 __fastcall sub_2352120(_BYTE *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = (unsigned __int8)a1[16];
  if ( (result & 2) != 0 )
    sub_23520B0(a1, a2);
  if ( (result & 1) != 0 )
  {
    v3 = *(_QWORD *)a1;
    if ( v3 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  }
  return result;
}
