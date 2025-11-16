// Function: sub_390C420
// Address: 0x390c420
//
__int64 __fastcall sub_390C420(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 result; // rax

  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 8) + 80LL))(
          *(_QWORD *)(a1 + 8),
          a2 + 128,
          *(_QWORD *)(a2 + 56)) )
    return 0;
  v4 = *(_QWORD *)(a2 + 88);
  v5 = v4 + 24LL * *(unsigned int *)(a2 + 96);
  if ( v4 == v5 )
    return 0;
  while ( 1 )
  {
    result = sub_390C370(a1, v4, a2, a3);
    if ( (_BYTE)result )
      break;
    v4 += 24;
    if ( v5 == v4 )
      return 0;
  }
  return result;
}
