// Function: sub_9D2250
// Address: 0x9d2250
//
__int64 __fastcall sub_9D2250(_BYTE *a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  result = (unsigned __int8)a1[8];
  if ( (result & 2) != 0 )
    sub_9D21E0(a1);
  if ( (result & 1) != 0 )
  {
    v2 = *(_QWORD *)a1;
    if ( v2 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  return result;
}
