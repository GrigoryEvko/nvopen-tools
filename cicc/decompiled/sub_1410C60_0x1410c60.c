// Function: sub_1410C60
// Address: 0x1410c60
//
__int64 __fastcall sub_1410C60(__int64 *a1, __int64 a2)
{
  _QWORD **v2; // rax
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r12
  _QWORD *v9; // rax
  char v11[16]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v12; // [rsp+10h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v2 = *(_QWORD ***)(a2 - 8);
  else
    v2 = (_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v3 = sub_1410110((__int64)a1, *v2);
  v6 = v3;
  v8 = v7;
  if ( !v7 || !v3 )
    return 0;
  v9 = sub_140D9A0(a1 + 3, *a1, a2, 1, v4, v5);
  v12 = 257;
  sub_140D830(a1 + 3, v8, (__int64)v9, (__int64)v11, 0, 0);
  return v6;
}
