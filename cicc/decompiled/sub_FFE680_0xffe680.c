// Function: sub_FFE680
// Address: 0xffe680
//
__int64 __fastcall sub_FFE680(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx

  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_QWORD *)(a2 - 32);
  if ( v2 == *(_QWORD *)a1 && v3 )
  {
    **(_QWORD **)(a1 + 8) = v3;
    return 1;
  }
  else if ( v2 && *(_QWORD *)a1 == v3 )
  {
    **(_QWORD **)(a1 + 8) = v2;
    return 1;
  }
  else
  {
    return 0;
  }
}
