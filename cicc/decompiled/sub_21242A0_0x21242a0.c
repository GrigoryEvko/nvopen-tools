// Function: sub_21242A0
// Address: 0x21242a0
//
__int64 *__fastcall sub_21242A0(__int64 a1, __int64 *a2)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v8; // rax
  __int128 v10; // [rsp-10h] [rbp-30h]

  v3 = sub_2120330(a1, *(_QWORD *)(a2[4] + 40), *(_QWORD *)(a2[4] + 48));
  v5 = v4;
  v7 = sub_2120330(a1, *(_QWORD *)(a2[4] + 80), *(_QWORD *)(a2[4] + 88));
  v8 = a2[4];
  if ( *(_QWORD *)(v8 + 40) == v3
    && *(_DWORD *)(v8 + 48) == (_DWORD)v5
    && *(_QWORD *)(v8 + 80) == v7
    && *(_DWORD *)(v8 + 88) == (_DWORD)v6 )
  {
    return 0;
  }
  *((_QWORD *)&v10 + 1) = v6;
  *(_QWORD *)&v10 = v7;
  return sub_1D2E2F0(*(_QWORD **)(a1 + 8), a2, *(_QWORD *)v8, *(_QWORD *)(v8 + 8), v3, v5, v10);
}
