// Function: sub_192D210
// Address: 0x192d210
//
__int64 __fastcall sub_192D210(_DWORD *a1, __int64 a2)
{
  if ( a1[2] == *(_DWORD *)(a2 + 8) )
    return (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)a1 + 16LL))(a1);
  else
    return 0;
}
