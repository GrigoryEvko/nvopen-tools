// Function: sub_295F380
// Address: 0x295f380
//
__int64 __fastcall sub_295F380(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r13
  _QWORD *v7; // r13
  __int64 v8; // rax
  unsigned int v10; // esi
  int v11; // eax
  int v12; // eax
  _QWORD *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-98h] BYREF
  void *v19; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v20[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  __int64 v22; // [rsp+30h] [rbp-70h]
  const char *v23; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v24[3]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v25; // [rsp+60h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD *)a1;
  v23 = ".us";
  LOWORD(v25) = 259;
  v5 = sub_F4B360(a2, v4, (__int64 *)&v23, v3, 0);
  sub_AA4AC0(v5, **(_QWORD **)(a1 + 8) + 24LL);
  sub_B1A4E0(*(_QWORD *)(a1 + 16), v5);
  v21 = a2;
  v6 = *(_QWORD *)a1;
  v20[0] = 2;
  v20[1] = 0;
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)v20);
  v22 = v6;
  v19 = &unk_49DD7B0;
  if ( (unsigned __int8)sub_F9E960(v6, (__int64)&v19, &v17) )
  {
    v7 = v17 + 5;
    goto LABEL_6;
  }
  v18 = v17;
  v10 = *(_DWORD *)(v6 + 24);
  v11 = *(_DWORD *)(v6 + 16);
  ++*(_QWORD *)v6;
  v12 = v11 + 1;
  if ( 4 * v12 >= 3 * v10 )
  {
    v10 *= 2;
    goto LABEL_27;
  }
  if ( v10 - *(_DWORD *)(v6 + 20) - v12 <= v10 >> 3 )
  {
LABEL_27:
    sub_CF32C0(v6, v10);
    sub_F9E960(v6, (__int64)&v19, &v18);
    v12 = *(_DWORD *)(v6 + 16) + 1;
  }
  v13 = v18;
  *(_DWORD *)(v6 + 16) = v12;
  v24[0] = 2;
  v24[1] = 0;
  v24[2] = -4096;
  v25 = 0;
  if ( v13[3] != -4096 )
    --*(_DWORD *)(v6 + 20);
  v23 = (const char *)&unk_49DB368;
  sub_D68D70(v24);
  v14 = v13[3];
  v15 = v21;
  if ( v14 != v21 )
  {
    if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
    {
      sub_BD60C0(v13 + 1);
      v15 = v21;
    }
    v13[3] = v15;
    if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
      sub_BD6050(v13 + 1, v20[0] & 0xFFFFFFFFFFFFFFF8LL);
  }
  v16 = v22;
  v13[5] = 6;
  v7 = v13 + 5;
  v13[6] = 0;
  v13[4] = v16;
  v13[7] = 0;
LABEL_6:
  v19 = &unk_49DB368;
  sub_D68D70(v20);
  v8 = v7[2];
  if ( v5 != v8 )
  {
    if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
      sub_BD60C0(v7);
    v7[2] = v5;
    if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
      sub_BD73F0((__int64)v7);
  }
  return v5;
}
