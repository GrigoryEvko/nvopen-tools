// Function: sub_2FCED00
// Address: 0x2fced00
//
__int64 __fastcall sub_2FCED00(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 (*v4)(void); // rax

  *a4 = 0;
  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  if ( v4 == sub_2E97330 )
    return 0;
  else
    return v4();
}
