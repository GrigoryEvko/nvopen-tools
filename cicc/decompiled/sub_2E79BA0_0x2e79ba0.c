// Function: sub_2E79BA0
// Address: 0x2e79ba0
//
__int64 __fastcall sub_2E79BA0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rsi
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 8) + 32 * a2;
  result = 0;
  if ( *(_DWORD *)(v3 + 24) < a3 )
  {
    *(_DWORD *)(v3 + 24) = a3;
    return 1;
  }
  return result;
}
