// Function: sub_2CAF420
// Address: 0x2caf420
//
__int64 __fastcall sub_2CAF420(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rdi
  _BOOL4 v9; // esi
  const char *v11; // [rsp+0h] [rbp-50h] BYREF
  char v12; // [rsp+20h] [rbp-30h]
  char v13; // [rsp+21h] [rbp-2Fh]

  v6 = sub_BCCE00(a1, 0x20u);
  if ( *(_DWORD *)(a2 + 112) != *(_DWORD *)(a3 + 12) )
    return sub_ACD640(v6, 0, 0);
  v7 = v6;
  v8 = *(_QWORD *)a3;
  v9 = *(_QWORD *)a2 == 0x200000002LL;
  v13 = 1;
  v11 = "idp.init";
  v12 = 3;
  return sub_2CAF330(v8, v9 + 1, v7, a4, (__int64)&v11);
}
