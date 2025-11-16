// Function: sub_892370
// Address: 0x892370
//
__int64 __fastcall sub_892370(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int64 i; // rax
  __int64 result; // rax
  __int64 v5; // r13

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_QWORD *)(*(_QWORD *)i + 96LL);
  if ( *(_QWORD *)(result + 72) )
  {
    v5 = *(_QWORD *)(result + 104);
    *a3 = sub_892330(a1);
    result = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 96) + 80LL) + 32LL);
    *a2 = result;
  }
  else
  {
    *a3 = 0;
    *a2 = 0;
  }
  return result;
}
