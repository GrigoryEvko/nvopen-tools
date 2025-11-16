// Function: sub_21239F0
// Address: 0x21239f0
//
__int64 *__fastcall sub_21239F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int128 v5; // rax
  __int64 v6; // rcx
  _QWORD *v7; // rdi
  __int64 v8; // r10
  __int64 v9; // r11

  v2 = sub_2120330(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v4 = v3;
  *(_QWORD *)&v5 = sub_2120330(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v6 = *(_QWORD *)(a2 + 32);
  if ( *(_QWORD *)(v6 + 40) == v2
    && *(_DWORD *)(v6 + 48) == (_DWORD)v4
    && *(_QWORD *)(v6 + 80) == (_QWORD)v5
    && *(_DWORD *)(v6 + 88) == DWORD2(v5) )
  {
    return 0;
  }
  v7 = *(_QWORD **)(a1 + 8);
  v8 = *(_QWORD *)v6;
  v9 = *(_QWORD *)(v6 + 8);
  if ( *(_DWORD *)(a2 + 56) == 3 )
    return sub_1D2E2F0(v7, (__int64 *)a2, v8, v9, v2, v4, v5);
  else
    return sub_1D2E330(v7, (__int64 *)a2, v8, v9, v2, v4, v5, *(_OWORD *)(v6 + 120));
}
