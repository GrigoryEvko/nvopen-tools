// Function: sub_134BF60
// Address: 0x134bf60
//
unsigned __int64 __fastcall sub_134BF60(__int64 a1)
{
  unsigned __int64 result; // rax
  int v2; // edx

  result = *(_QWORD *)(a1 + 5264);
  if ( result )
  {
    v2 = 64;
LABEL_5:
    _BitScanReverse64(&result, result);
    return *(_QWORD *)(a1 + 8LL * (unsigned int)(v2 + result) + 4232);
  }
  result = *(_QWORD *)(a1 + 5256);
  if ( result )
  {
    v2 = 0;
    goto LABEL_5;
  }
  return result;
}
