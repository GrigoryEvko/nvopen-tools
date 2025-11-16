// Function: sub_2352440
// Address: 0x2352440
//
__int64 __fastcall sub_2352440(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = *((unsigned __int8 *)a1 + 40);
  if ( (result & 2) != 0 )
    sub_23523D0(a1, a2);
  if ( (result & 1) == 0 )
    return sub_C7D6A0(a1[2], 4LL * *((unsigned int *)a1 + 8), 4);
  v3 = *a1;
  if ( v3 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  return result;
}
