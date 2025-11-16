// Function: sub_1E6A620
// Address: 0x1e6a620
//
__int64 __fastcall sub_1E6A620(_QWORD *a1)
{
  __int64 (*v1)(); // rax
  __int64 v2; // rax

  if ( *((_BYTE *)a1 + 152) )
    return a1[20];
  v1 = *(__int64 (**)())(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v1 == sub_1D00B10 )
    BUG();
  v2 = v1();
  return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v2 + 24LL))(v2, *a1);
}
