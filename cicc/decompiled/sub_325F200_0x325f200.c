// Function: sub_325F200
// Address: 0x325f200
//
__int64 __fastcall sub_325F200(unsigned int *a1, __int64 a2)
{
  if ( *(_DWORD *)(a2 + 24) == 159 && *(_DWORD *)(a2 + 64) == 2 )
    return *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * *a1);
  else
    return 0;
}
