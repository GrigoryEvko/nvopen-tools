// Function: sub_1802170
// Address: 0x1802170
//
_QWORD *__fastcall sub_1802170(__int64 a1, __int64 a2, __int64 *a3, _BYTE *a4, __int64 a5)
{
  __int64 v7; // r15
  char v8; // r8
  _QWORD *v9; // r12
  int v10; // eax
  size_t v11; // rdx
  char *v12; // rsi
  char v14; // [rsp+Ch] [rbp-64h]
  _QWORD v15[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v16[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v17; // [rsp+30h] [rbp-40h]

  v7 = *a3;
  v8 = (*(_DWORD *)(a1 + 276) != 3) + 7;
  if ( a5 && *a4 == 1 )
  {
    --a5;
    ++a4;
  }
  v15[1] = a5;
  v16[0] = "__asan_global_";
  v16[1] = v15;
  v14 = v8;
  v15[0] = a4;
  v17 = 1283;
  v9 = sub_1648A60(88, 1u);
  if ( v9 )
    sub_15E51E0((__int64)v9, a2, v7, 0, v14, (__int64)a3, (__int64)v16, 0, 0, 0, 0);
  v10 = *(_DWORD *)(a1 + 276);
  v11 = 12;
  v12 = "asan_globals";
  if ( v10 != 2 )
  {
    v11 = 8;
    if ( v10 == 3 )
      v11 = 29;
    v12 = ".ASAN$GL";
    if ( v10 == 3 )
      v12 = "__DATA,__asan_globals,regular";
  }
  sub_15E5D20((__int64)v9, v12, v11);
  return v9;
}
