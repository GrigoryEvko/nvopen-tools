// Function: sub_250D1D0
// Address: 0x250d1d0
//
__int64 __fastcall sub_250D1D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 24LL))(v2) )
    return 1;
  else
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 104LL))(a1, a2);
}
