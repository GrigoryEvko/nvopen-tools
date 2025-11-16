// Function: sub_E9A5B0
// Address: 0xe9a5b0
//
__int64 __fastcall sub_E9A5B0(__int64 a1, unsigned __int8 *a2)
{
  __int64 (__fastcall *v2)(__int64, unsigned __int8 *); // rax

  v2 = *(__int64 (__fastcall **)(__int64, unsigned __int8 *))(*(_QWORD *)a1 + 528LL);
  if ( v2 == sub_E9A480 )
    return sub_E9A370(a1, a2);
  else
    return v2(a1, a2);
}
