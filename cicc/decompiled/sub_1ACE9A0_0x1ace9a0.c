// Function: sub_1ACE9A0
// Address: 0x1ace9a0
//
__int64 __fastcall sub_1ACE9A0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rbx
  int v6; // eax
  __int64 *v7; // rax
  __int64 v8; // rbx
  const char *v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // rdx
  char *v13; // rsi

  if ( (*(_BYTE *)(a3 + 32) & 0xFu) - 7 <= 1 && (a4 || *(_QWORD *)(a2 + 16)) )
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = sub_16D1B30(
           (__int64 *)(v5 + 48),
           *(unsigned __int8 **)(*(_QWORD *)(a3 + 40) + 176LL),
           *(_QWORD *)(*(_QWORD *)(a3 + 40) + 184LL));
    if ( v6 == -1 )
      v7 = (__int64 *)(*(_QWORD *)(v5 + 48) + 8LL * *(unsigned int *)(v5 + 56));
    else
      v7 = (__int64 *)(*(_QWORD *)(v5 + 48) + 8LL * v6);
    v8 = *v7;
    v9 = sub_1649960(a3);
    sub_1ACE1F0(a1, v9, v10, *(_DWORD *)(v8 + 16), *(_DWORD *)(v8 + 20));
    return a1;
  }
  else
  {
    v13 = (char *)sub_1649960(a3);
    *(_QWORD *)a1 = a1 + 16;
    if ( v13 )
    {
      sub_1ACE140((__int64 *)a1, v13, (__int64)&v13[v12]);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
    }
    return a1;
  }
}
