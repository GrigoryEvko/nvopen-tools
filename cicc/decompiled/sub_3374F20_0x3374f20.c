// Function: sub_3374F20
// Address: 0x3374f20
//
__int64 __fastcall sub_3374F20(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r13
  __int64 result; // rax

  if ( a2 )
  {
    v4 = *(_QWORD *)(a1 + 864);
    nullsub_1875(a2, v4, 0);
    *(_QWORD *)(v4 + 384) = a2;
    *(_DWORD *)(v4 + 392) = a3;
    return sub_33E2B60(v4, 0);
  }
  else
  {
    *(_BYTE *)(a1 + 1016) = 1;
  }
  return result;
}
