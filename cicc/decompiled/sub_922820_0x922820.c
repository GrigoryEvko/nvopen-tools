// Function: sub_922820
// Address: 0x922820
//
__int64 __fastcall sub_922820(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  _BOOL4 v5; // r15d
  int v6; // eax
  __int64 v7; // rdi
  int v8; // ebx
  __int64 v9; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a3 + 56);
  if ( (*(_BYTE *)(v4 + 89) & 1) != 0 || *(char *)(v4 + 169) < 0 )
  {
    v5 = 0;
    v11 = *(_QWORD *)(v4 + 120);
    v6 = sub_91CB50(*(_QWORD *)(a3 + 56), a2, a3);
    v7 = *(_QWORD *)(v4 + 120);
    v8 = v6;
    if ( (*(_BYTE *)(v7 + 140) & 0xFB) == 8 )
      v5 = (sub_8D4C10(v7, dword_4F077C4 != 2) & 2) != 0;
    v9 = sub_9227D0(a2, v4);
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v9;
    *(_DWORD *)(a1 + 48) = v5;
    *(_QWORD *)(a1 + 16) = v11;
    *(_DWORD *)(a1 + 24) = v8;
  }
  else
  {
    sub_922290(a1, a2, *(_QWORD *)(a3 + 56));
  }
  return a1;
}
