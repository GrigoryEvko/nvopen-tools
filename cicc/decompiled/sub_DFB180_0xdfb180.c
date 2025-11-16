// Function: sub_DFB180
// Address: 0xdfb180
//
__int64 __fastcall sub_DFB180(__int64 *a1, unsigned __int8 a2)
{
  __int64 v2; // rdi
  __int64 (__fastcall *v3)(__int64, unsigned __int8); // rax

  v2 = *a1;
  v3 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v2 + 992LL);
  if ( v3 == sub_DF5FD0 )
    return a2;
  else
    return v3(v2, a2);
}
