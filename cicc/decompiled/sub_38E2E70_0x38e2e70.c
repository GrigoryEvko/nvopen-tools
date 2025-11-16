// Function: sub_38E2E70
// Address: 0x38e2e70
//
__int64 __fastcall sub_38E2E70(__int64 a1, unsigned __int64 a2, int a3)
{
  _QWORD *v3; // rcx

  if ( !a3 )
    a3 = sub_16CE270(*(__int64 **)(a1 + 344), a2);
  v3 = *(_QWORD **)(a1 + 344);
  *(_DWORD *)(a1 + 376) = a3;
  return sub_392A730(
           a1 + 144,
           *(_QWORD *)(*(_QWORD *)(*v3 + 24LL * (unsigned int)(a3 - 1)) + 8LL),
           *(_QWORD *)(*(_QWORD *)(*v3 + 24LL * (unsigned int)(a3 - 1)) + 16LL)
         - *(_QWORD *)(*(_QWORD *)(*v3 + 24LL * (unsigned int)(a3 - 1)) + 8LL),
           a2);
}
