// Function: sub_134BFB0
// Address: 0x134bfb0
//
__int64 __fastcall sub_134BFB0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax

  v2 = a2;
  *(_BYTE *)(a2 + 36) = 1;
  result = sub_134B960(a1, a2);
  if ( !*(_BYTE *)(a2 + 17) )
  {
    if ( !*(_BYTE *)(a2 + 19) )
      goto LABEL_3;
LABEL_6:
    result = sub_134B5B0((__int64)a1, a2);
    if ( !*(_BYTE *)(a2 + 20) )
      return result;
    goto LABEL_7;
  }
  result = sub_134B7F0((__int64)a1, a2);
  if ( *(_BYTE *)(a2 + 19) )
    goto LABEL_6;
LABEL_3:
  if ( !*(_BYTE *)(a2 + 20) )
    return result;
LABEL_7:
  *(_BYTE *)(a2 + 32) = 1;
  *(_QWORD *)(a2 + 80) = a2;
  *(_QWORD *)(a2 + 88) = a2;
  result = a1[659];
  if ( result )
  {
    *(_QWORD *)(a2 + 80) = *(_QWORD *)(result + 88);
    *(_QWORD *)(a1[659] + 88LL) = a2;
    *(_QWORD *)(a2 + 88) = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 80LL);
    *(_QWORD *)(*(_QWORD *)(a1[659] + 88LL) + 80LL) = a1[659];
    result = *(_QWORD *)(a2 + 88);
    *(_QWORD *)(result + 80) = a2;
    v2 = *(_QWORD *)(a2 + 80);
  }
  a1[659] = v2;
  return result;
}
