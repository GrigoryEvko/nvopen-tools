// Function: sub_942D70
// Address: 0x942d70
//
__int64 __fastcall sub_942D70(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // r12
  int v4; // r14d
  int v5; // r13d
  int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // r12
  _QWORD *v10; // [rsp+10h] [rbp-60h]
  __int64 v11; // [rsp+18h] [rbp-58h]
  _QWORD v12[10]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a2 + 160);
  v3 = 8LL * *(_QWORD *)(v2 + 128);
  if ( *(char *)(v2 + 142) >= 0 && *(_BYTE *)(v2 + 140) == 12 )
    v4 = 8 * sub_8D4AB0(v2);
  else
    v4 = 8 * *(_DWORD *)(v2 + 136);
  v5 = sub_941B90(a1, v2);
  v10 = v12;
  v11 = 0x400000001LL;
  v12[0] = sub_ADD550(a1 + 16, 0, 1);
  v6 = sub_ADCD70(a1 + 16, v12, 1);
  v7 = v3;
  v8 = sub_ADE2A0((int)a1 + 16, v3, v4, v5, v6, 0, 0, 0, 0);
  if ( v10 != v12 )
    _libc_free(v10, v7);
  return v8;
}
