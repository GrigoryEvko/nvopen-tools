// Function: sub_1DBCA20
// Address: 0x1dbca20
//
__int64 __fastcall sub_1DBCA20(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 result; // rax

  v2 = **(_QWORD **)a2;
  if ( (v2 & 6) == 0 )
    return 0;
  v3 = *(_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8) - 16);
  if ( (v3 & 6) == 0 )
    return 0;
  v4 = *(_QWORD *)(a1 + 272);
  v5 = sub_1DA9310(v4, v2);
  result = sub_1DA9310(v4, v3);
  if ( v5 != result )
    return 0;
  return result;
}
