// Function: sub_120E240
// Address: 0x120e240
//
__int64 __fastcall sub_120E240(__int64 a1, _BYTE *a2)
{
  unsigned int v2; // r13d
  __int64 v4; // r15
  int v7; // eax
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rsi
  int v11; // eax
  _QWORD *v12; // rdi
  __int64 v13; // rsi
  char v14; // al
  _QWORD *v15; // rdi
  unsigned __int64 v16; // [rsp+8h] [rbp-88h]
  _QWORD *v17; // [rsp+10h] [rbp-80h] BYREF
  __int64 v18; // [rsp+18h] [rbp-78h]
  _QWORD v19[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v20[4]; // [rsp+30h] [rbp-60h] BYREF
  char v21; // [rsp+50h] [rbp-40h]
  char v22; // [rsp+51h] [rbp-3Fh]

  v2 = 0;
  *a2 = 1;
  if ( *(_DWORD *)(a1 + 240) != 76 )
    return v2;
  v4 = a1 + 176;
  v7 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v7;
  if ( v7 != 12 )
  {
    v22 = 1;
    v8 = *(_QWORD *)(a1 + 232);
    v21 = 3;
    v20[0] = "Expected '(' in syncscope";
    sub_11FD800(v4, v8, (__int64)v20, 1);
    return 1;
  }
  v17 = v19;
  *(_DWORD *)(a1 + 240) = sub_1205200(v4);
  v9 = *(_QWORD *)(a1 + 232);
  v18 = 0;
  LOBYTE(v19[0]) = 0;
  v16 = v9;
  v2 = sub_120B3D0(a1, (__int64)&v17);
  if ( (_BYTE)v2 )
  {
    v22 = 1;
    v21 = 3;
    v20[0] = "Expected synchronization scope name";
    sub_11FD800(v4, v16, (__int64)v20, 1);
LABEL_9:
    if ( v17 != v19 )
      j_j___libc_free_0(v17, v19[0] + 1LL);
    return 1;
  }
  v10 = *(_QWORD *)(a1 + 232);
  if ( *(_DWORD *)(a1 + 240) != 13 )
  {
    v22 = 1;
    v21 = 3;
    v20[0] = "Expected ')' in syncscope";
    sub_11FD800(v4, v10, (__int64)v20, 1);
    goto LABEL_9;
  }
  v11 = sub_1205200(v4);
  v12 = *(_QWORD **)a1;
  v13 = (__int64)v17;
  *(_DWORD *)(a1 + 240) = v11;
  v14 = sub_B6F810(v12, v13, v18);
  v15 = v17;
  *a2 = v14;
  if ( v15 != v19 )
    j_j___libc_free_0(v15, v19[0] + 1LL);
  return v2;
}
