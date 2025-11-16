// Function: sub_26F8EA0
// Address: 0x26f8ea0
//
__int64 __fastcall sub_26F8EA0(__int64 a1)
{
  unsigned int v1; // r13d
  _BYTE *v2; // rsi
  __int64 v3; // rdx
  int v4; // ecx
  _QWORD *v6; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v8; // [rsp+20h] [rbp-40h]
  __int64 v9; // [rsp+28h] [rbp-38h]
  __int64 v10; // [rsp+30h] [rbp-30h]

  v1 = 0;
  v2 = *(_BYTE **)(a1 + 232);
  v3 = *(_QWORD *)(a1 + 240);
  v6 = v7;
  sub_26F69E0((__int64 *)&v6, v2, (__int64)&v2[v3]);
  v4 = *(_DWORD *)(a1 + 284);
  v8 = *(_QWORD *)(a1 + 264);
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_QWORD *)(a1 + 280);
  if ( (unsigned int)(v8 - 38) <= 1 )
    LOBYTE(v1) = v4 == 3;
  if ( v6 != v7 )
    j_j___libc_free_0((unsigned __int64)v6);
  return v1;
}
