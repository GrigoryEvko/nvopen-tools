// Function: sub_2351C10
// Address: 0x2351c10
//
__int64 __fastcall sub_2351C10(_BYTE *a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  result = (unsigned __int8)a1[8];
  if ( (result & 2) != 0 )
    sub_A05710(a1);
  if ( (result & 1) != 0 )
  {
    v2 = *(_QWORD *)a1;
    if ( v2 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  return result;
}
