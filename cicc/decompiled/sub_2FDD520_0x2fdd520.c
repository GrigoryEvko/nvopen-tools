// Function: sub_2FDD520
// Address: 0x2fdd520
//
unsigned __int64 __fastcall sub_2FDD520(__int64 a1, __int64 a2)
{
  __int16 v2; // ax
  int v3; // eax
  __int64 v4; // rbx
  __int64 v6; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int8 v7; // [rsp+8h] [rbp-18h]

  v2 = *(_WORD *)(a2 + 68);
  switch ( v2 )
  {
    case 28:
      sub_2FC8700((__int64)&v6, a2);
      return (unsigned __int64)(*(_DWORD *)(*(_QWORD *)(v6 + 32) + 40LL * ((unsigned int)v7 + 3) + 24)
                              + (unsigned int)v7
                              + 5) << 32;
    case 32:
      v3 = sub_2E88FE0(a2) + *(unsigned __int8 *)(*(_QWORD *)(a2 + 16) + 9LL);
      v4 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (unsigned int)(v3 + 2) + 24) + v3 + 4);
      return (v4 << 32) | (*(unsigned __int8 *)(*(_QWORD *)(a2 + 16) + 9LL) + (unsigned int)sub_2E88FE0(a2));
    case 26:
      sub_2FC86F0(&v6, a2);
      return 0x200000000LL;
    default:
      BUG();
  }
}
