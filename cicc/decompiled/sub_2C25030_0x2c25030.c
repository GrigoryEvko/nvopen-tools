// Function: sub_2C25030
// Address: 0x2c25030
//
char __fastcall sub_2C25030(_QWORD *a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  _BYTE *v6; // r12
  char result; // al

  if ( !a2 )
    BUG();
  if ( *a1 == a2[5] )
    return 1;
  v4 = *a2;
  v5 = a1[2];
  if ( v5 )
    v5 += 96;
  v6 = (_BYTE *)a1[1];
  *v6 = (*(__int64 (__fastcall **)(__int64 *, __int64))(v4 + 24))(a2, v5);
  result = *(_BYTE *)a1[1];
  if ( result )
    return *(_BYTE *)(a1[2] + 8LL) == 9;
  return result;
}
