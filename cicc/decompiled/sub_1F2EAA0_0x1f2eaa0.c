// Function: sub_1F2EAA0
// Address: 0x1f2eaa0
//
__int64 __fastcall sub_1F2EAA0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 (*v4)(void); // rax

  *a4 = 0;
  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 48LL);
  if ( v4 == sub_1E1C810 )
    return 0;
  else
    return v4();
}
