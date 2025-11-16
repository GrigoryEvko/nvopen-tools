// Function: sub_CA53E0
// Address: 0xca53e0
//
char __fastcall sub_CA53E0(__int64 a1)
{
  if ( !qword_4F84FB0 )
    sub_C7D570(&qword_4F84FB0, sub_CA5520, (__int64)sub_CA53C0);
  if ( !*(_DWORD *)(qword_4F84FB0 + 136) )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1);
  if ( !qword_4F84FB0 )
    sub_C7D570(&qword_4F84FB0, sub_CA5520, (__int64)sub_CA53C0);
  return *(_DWORD *)(qword_4F84FB0 + 136) == 1;
}
