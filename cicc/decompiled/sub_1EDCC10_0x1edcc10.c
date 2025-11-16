// Function: sub_1EDCC10
// Address: 0x1edcc10
//
void __fastcall sub_1EDCC10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // r12
  int v7; // eax
  __int64 *v8; // rdx
  __int64 *v9; // rsi
  int v10; // r9d
  __int64 v11; // r12
  unsigned __int64 v12[2]; // [rsp+0h] [rbp-A0h] BYREF
  _BYTE v13[48]; // [rsp+10h] [rbp-90h] BYREF
  _BYTE *v14; // [rsp+40h] [rbp-60h]
  __int64 v15; // [rsp+48h] [rbp-58h]
  _BYTE v16[16]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v17; // [rsp+60h] [rbp-40h]

  v6 = *a1;
  v7 = *(_DWORD *)(a2 + 8);
  v8 = *(__int64 **)(*a1 + 8);
  v9 = *(__int64 **)(*a1 + 16);
  if ( v7 )
  {
    v17 = 0;
    v12[1] = 0x200000000LL;
    v15 = 0x200000000LL;
    v12[0] = (unsigned __int64)v13;
    v14 = v16;
    sub_1EDCA90((__int64)v12, v9, v8, a4, a5);
    sub_1EDBA90(*(_QWORD *)v6, a2, (__int64)v12, *(_DWORD *)(a2 + 112), *(_DWORD **)(v6 + 24), v10);
    v11 = v17;
    if ( v17 )
    {
      sub_1ED8B20(*(_QWORD *)(v17 + 16));
      j_j___libc_free_0(v11, 48);
    }
    if ( v14 != v16 )
      _libc_free((unsigned __int64)v14);
    if ( (_BYTE *)v12[0] != v13 )
      _libc_free(v12[0]);
  }
  else
  {
    sub_1EDCA90(a2, v9, v8, a4, a5);
  }
}
