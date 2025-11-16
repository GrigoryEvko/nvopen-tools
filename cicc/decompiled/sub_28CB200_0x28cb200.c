// Function: sub_28CB200
// Address: 0x28cb200
//
__int64 __fastcall sub_28CB200(_QWORD *a1)
{
  if ( !*((_DWORD *)a1 + 4) )
    a1[2] = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 32LL))(a1);
  return a1[2];
}
