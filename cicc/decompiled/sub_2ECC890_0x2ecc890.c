// Function: sub_2ECC890
// Address: 0x2ecc890
//
__int64 __fastcall sub_2ECC890(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax
  __int64 result; // rax

  v2 = a1[4];
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 48LL);
  if ( v3 == sub_23CE2B0 )
    return sub_2ECC540(a1);
  result = ((__int64 (__fastcall *)(__int64, _QWORD *))v3)(v2, a1);
  if ( !result )
    return sub_2ECC540(a1);
  return result;
}
