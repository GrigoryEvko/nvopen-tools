// Function: sub_EA24B0
// Address: 0xea24b0
//
__int64 __fastcall sub_EA24B0(__int64 a1, unsigned __int64 a2, int a3)
{
  _QWORD *v3; // rcx

  if ( !a3 )
    a3 = sub_C8ED90(*(__int64 **)(a1 + 248), a2);
  v3 = *(_QWORD **)(a1 + 248);
  *(_DWORD *)(a1 + 304) = a3;
  return sub_1095BD0(
           a1 + 40,
           *(_QWORD *)(*(_QWORD *)(*v3 + 24LL * (unsigned int)(a3 - 1)) + 8LL),
           *(_QWORD *)(*(_QWORD *)(*v3 + 24LL * (unsigned int)(a3 - 1)) + 16LL)
         - *(_QWORD *)(*(_QWORD *)(*v3 + 24LL * (unsigned int)(a3 - 1)) + 8LL),
           a2,
           1);
}
