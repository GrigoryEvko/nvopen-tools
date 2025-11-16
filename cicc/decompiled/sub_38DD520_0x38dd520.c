// Function: sub_38DD520
// Address: 0x38dd520
//
__int64 __fastcall sub_38DD520(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 (*v4)(); // rdx

  result = sub_38DD280(a1, a2);
  if ( result )
  {
    v3 = result;
    v4 = *(__int64 (**)())(*(_QWORD *)a1 + 16LL);
    result = 1;
    if ( v4 != sub_38DBC10 )
      result = ((__int64 (__fastcall *)(__int64))v4)(a1);
    *(_QWORD *)(v3 + 32) = result;
  }
  return result;
}
