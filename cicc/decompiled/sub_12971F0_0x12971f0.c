// Function: sub_12971F0
// Address: 0x12971f0
//
__int64 __fastcall sub_12971F0(__int64 a1, unsigned __int64 a2, int a3)
{
  int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rax

  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v4 = sub_8D4AB0(a2);
  else
    v4 = *(_DWORD *)(a2 + 136);
  v5 = sub_127A040(*(_QWORD *)(a1 + 32) + 8LL, a2);
  if ( v4 == (unsigned int)sub_15A9FE0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 368LL), v5) )
    return 0;
  v6 = sub_1643350(*(_QWORD *)(a1 + 40));
  return sub_159C470(v6, (a3 << 16) | (unsigned int)(unsigned __int16)v4, 0);
}
