// Function: sub_3365B50
// Address: 0x3365b50
//
__int64 __fastcall sub_3365B50(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  if ( a2 )
  {
    nullsub_1875(a2, a1, 0);
    *(_QWORD *)(a1 + 384) = a2;
    *(_DWORD *)(a1 + 392) = a3;
    return sub_33E2B60(a1, 0);
  }
  else
  {
    *(_QWORD *)(a1 + 384) = 0;
    *(_DWORD *)(a1 + 392) = a3;
  }
  return result;
}
