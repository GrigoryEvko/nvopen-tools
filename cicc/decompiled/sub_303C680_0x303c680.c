// Function: sub_303C680
// Address: 0x303c680
//
__int64 __fastcall sub_303C680(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int16 *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 v10; // [rsp-38h] [rbp-38h] BYREF
  int v11; // [rsp-30h] [rbp-30h]

  if ( !*(_DWORD *)(a2 + 96) || !*(_DWORD *)(a2 + 100) )
    return a2;
  v6 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v10 = 0;
  v11 = 0;
  v9 = sub_33F17F0(a4, 51, &v10, v7, v8);
  if ( v10 )
    sub_B91220((__int64)&v10, v10);
  return v9;
}
