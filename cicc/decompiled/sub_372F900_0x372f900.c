// Function: sub_372F900
// Address: 0x372f900
//
__int64 __fastcall sub_372F900(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 v6; // rdx
  char *v8; // [rsp+0h] [rbp-40h] BYREF
  char v9; // [rsp+20h] [rbp-20h]
  char v10; // [rsp+21h] [rbp-1Fh]

  v5 = *(unsigned int *)(a1 + 152);
  v6 = *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 32 * v6 - 16) == v5 )
  {
    *(_DWORD *)(a1 + 8) = v6 - 1;
    return 0;
  }
  else
  {
    v10 = 1;
    v8 = "debug_loc";
    v9 = 3;
    *(_QWORD *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8) - 24) = sub_31DCC50(a2, (__int64 *)&v8, v6, v5, a5);
    return 1;
  }
}
