// Function: sub_A68C30
// Address: 0xa68c30
//
void __fastcall sub_A68C30(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // [rsp-8h] [rbp-4F8h]
  _QWORD v15[2]; // [rsp+10h] [rbp-4E0h] BYREF
  __int64 v16; // [rsp+20h] [rbp-4D0h]
  __int64 v17; // [rsp+28h] [rbp-4C8h]
  __int64 v18; // [rsp+30h] [rbp-4C0h]
  int v19; // [rsp+3Ch] [rbp-4B4h]
  __int64 v20; // [rsp+40h] [rbp-4B0h]
  char *v21; // [rsp+58h] [rbp-498h]
  char v22; // [rsp+70h] [rbp-480h] BYREF
  _BYTE v23[400]; // [rsp+80h] [rbp-470h] BYREF
  __int64 v24[92]; // [rsp+210h] [rbp-2E0h] BYREF

  sub_A55A10((__int64)v23, *(_QWORD *)(a1 + 40), 0);
  sub_A54BD0((__int64)v15, a2);
  sub_A685A0((__int64)v24, (__int64)v15, (__int64)v23, *(_QWORD *)(a1 + 40), a3, a5, a4);
  sub_A65640(v24, a1);
  sub_A555E0((__int64)v24);
  v6 = v12;
  v15[0] = &unk_49DC840;
  if ( v18 != v16 )
    sub_CB5AE0(v15);
  v7 = v20;
  if ( v20 )
  {
    if ( !v19 || v16 )
    {
      v8 = v17 - v16;
    }
    else
    {
      v11 = sub_CB54F0(v15);
      v7 = v20;
      v8 = v11;
    }
    v9 = *(_QWORD *)(v7 + 32);
    v10 = *(_QWORD *)(v7 + 16);
    if ( v8 )
    {
      if ( v9 != v10 )
        sub_CB5AE0(v7);
      v6 = sub_2207820(v8);
      sub_CB5980(v7, v6, v8, 1);
    }
    else
    {
      if ( v9 != v10 )
        sub_CB5AE0(v7);
      v6 = 0;
      sub_CB5980(v7, 0, 0, 0);
    }
  }
  if ( v21 != &v22 )
    _libc_free(v21, v6);
  sub_CB5840(v15);
  sub_A552A0((__int64)v23, v6);
}
