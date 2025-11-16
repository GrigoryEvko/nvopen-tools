// Function: sub_217DBB0
// Address: 0x217dbb0
//
__int64 __fastcall sub_217DBB0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 17) & 1LL;
  if ( a2 && (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) & 0x20000LL) != 0 )
  {
    if ( *(_DWORD *)(a1 + 40) > 2u && (v3 = *(_QWORD *)(a1 + 32), *(_BYTE *)(v3 + 80) == 1) )
      *a2 = *(_QWORD *)(v3 + 104);
    else
      *a2 = 0;
  }
  return result;
}
