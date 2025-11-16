// Function: sub_2ECE2C0
// Address: 0x2ece2c0
//
__int64 __fastcall sub_2ECE2C0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 result; // rax
  bool v10; // zf

  a1[17] = a2;
  a1[2] = a2 + 600;
  a1[3] = *(_QWORD *)(a2 + 24);
  sub_2EC8A00((__int64)(a1 + 5), a2, a2 + 600);
  sub_2ECDD50((__int64)(a1 + 18), a1[17], a1[2], (__int64)(a1 + 5), v3, v4);
  sub_2ECDD50((__int64)(a1 + 108), a1[17], a1[2], (__int64)(a1 + 5), v5, v6);
  v7 = a1[2];
  v8 = v7 + 80;
  result = sub_2FF7B90(v7);
  if ( !(_BYTE)result )
    v8 = 0;
  if ( a1[37] )
  {
    if ( a1[127] )
      return result;
LABEL_7:
    result = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1[17] + 16) + 1040LL))(
               *(_QWORD *)(a1[17] + 16),
               v8);
    a1[127] = result;
    return result;
  }
  result = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1[17] + 16) + 1040LL))(
             *(_QWORD *)(a1[17] + 16),
             v8);
  v10 = a1[127] == 0;
  a1[37] = result;
  if ( v10 )
    goto LABEL_7;
  return result;
}
