// Function: sub_2FCED30
// Address: 0x2fced30
//
__int64 __fastcall sub_2FCED30(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 (*v4)(void); // rax

  *a4 = 0;
  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 120LL);
  if ( v4 == sub_2F4C0B0 )
    return 0;
  else
    return v4();
}
