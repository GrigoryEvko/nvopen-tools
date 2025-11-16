// Function: sub_222AF60
// Address: 0x222af60
//
__int64 __fastcall sub_222AF60(__int64 a1, __int64 a2)
{
  FILE **v2; // rbp
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // r12
  __off_t v8; // rbp

  if ( (*(_BYTE *)(a1 + 120) & 8) == 0 )
    return -1;
  v2 = (FILE **)(a1 + 104);
  if ( !sub_2207CD0((_QWORD *)(a1 + 104)) )
    return -1;
  v6 = *(_QWORD *)(a1 + 200);
  v7 = *(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16);
  if ( !v6 )
    sub_426219(0, a2, v4, v5);
  if ( (*(int (__fastcall **)(__int64))(*(_QWORD *)v6 + 40LL))(v6) >= 0 )
  {
    v8 = sub_2207F70(v2);
    v7 += v8 / (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 64LL))(*(_QWORD *)(a1 + 200));
  }
  return v7;
}
