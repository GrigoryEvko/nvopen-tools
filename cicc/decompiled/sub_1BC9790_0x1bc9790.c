// Function: sub_1BC9790
// Address: 0x1bc9790
//
__int64 __fastcall sub_1BC9790(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 result; // rax
  _BYTE v4[33]; // [rsp+Fh] [rbp-21h] BYREF

  v1 = *(_QWORD *)(a1 + 184);
  while ( *(_QWORD *)(a1 + 192) != v1 )
  {
    sub_1BC9610(a1, v1, (void (__fastcall *)(__int64, __int64))sub_1BB9460, (__int64)v4);
    v2 = *(_QWORD *)(v1 + 32);
    result = *(_QWORD *)(v1 + 40) + 40LL;
    if ( v2 == result || !v2 )
      v1 = 0;
    else
      v1 = v2 - 24;
  }
  *(_DWORD *)(a1 + 112) = 0;
  return result;
}
