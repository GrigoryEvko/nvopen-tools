// Function: sub_E5E420
// Address: 0xe5e420
//
__int64 __fastcall sub_E5E420(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  __int64 result; // rax
  __int64 v6; // rbx
  __int64 v7; // r13

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 120LL);
  if ( v4 == sub_E5B830 )
    return 0;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v4)(v3, a2 + 112, *(_QWORD *)(a2 + 32)) )
    return 0;
  v6 = *(_QWORD *)(a2 + 72);
  v7 = v6 + 24LL * *(unsigned int *)(a2 + 80);
  if ( v6 == v7 )
    return 0;
  while ( 1 )
  {
    result = sub_E5E370(a1, v6, a2);
    if ( (_BYTE)result )
      break;
    v6 += 24;
    if ( v7 == v6 )
      return 0;
  }
  return result;
}
