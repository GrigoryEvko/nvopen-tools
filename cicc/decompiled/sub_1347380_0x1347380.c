// Function: sub_1347380
// Address: 0x1347380
//
__int64 __fastcall sub_1347380(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // [rsp+8h] [rbp-18h] BYREF

  if ( (*(_DWORD *)(a2 + 32) & 0xFFFF00) != 0 )
  {
    *(_WORD *)(a2 + 19) = 0;
    return 0;
  }
  else
  {
    result = *(_QWORD *)(a2 + 104);
    *(_BYTE *)(a2 + 19) = *(_QWORD *)(a2 + 176) != result;
    if ( (unsigned __int64)(result << 12) >= *(_QWORD *)(a1 + 5632) && !*(_BYTE *)(a2 + 16) )
    {
      (*(void (__fastcall **)(__int64 *, __int64))(*(_QWORD *)(a1 + 56) + 296LL))(&v4, 1);
      v3 = v4;
      *(_BYTE *)(a2 + 20) = 1;
      *(_QWORD *)(a2 + 24) = v3;
      result = *(_QWORD *)(a2 + 104);
    }
    if ( !result )
      *(_BYTE *)(a2 + 20) = 0;
  }
  return result;
}
