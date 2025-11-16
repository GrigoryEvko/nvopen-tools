// Function: sub_A69980
// Address: 0xa69980
//
void __fastcall sub_A69980(__int64 (__fastcall **a1)(), __int64 a2, __int64 a3, char a4, char a5, __m128i a6)
{
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // [rsp-8h] [rbp-4F8h]
  _QWORD v16[2]; // [rsp+10h] [rbp-4E0h] BYREF
  __int64 v17; // [rsp+20h] [rbp-4D0h]
  __int64 v18; // [rsp+28h] [rbp-4C8h]
  __int64 v19; // [rsp+30h] [rbp-4C0h]
  int v20; // [rsp+3Ch] [rbp-4B4h]
  __int64 v21; // [rsp+40h] [rbp-4B0h]
  char *v22; // [rsp+58h] [rbp-498h]
  char v23; // [rsp+70h] [rbp-480h] BYREF
  _BYTE v24[400]; // [rsp+80h] [rbp-470h] BYREF
  _BYTE v25[736]; // [rsp+210h] [rbp-2E0h] BYREF

  sub_A55A10((__int64)v24, (__int64)a1, 0);
  sub_A54BD0((__int64)v16, a2);
  sub_A685A0((__int64)v25, (__int64)v16, (__int64)v24, (__int64)a1, a3, a5, a4);
  sub_A66340((__int64)v25, a1, a6);
  sub_A555E0((__int64)v25);
  v7 = v13;
  v16[0] = &unk_49DC840;
  if ( v19 != v17 )
    sub_CB5AE0(v16);
  v8 = v21;
  if ( v21 )
  {
    if ( !v20 || v17 )
    {
      v9 = v18 - v17;
    }
    else
    {
      v12 = sub_CB54F0(v16);
      v8 = v21;
      v9 = v12;
    }
    v10 = *(_QWORD *)(v8 + 32);
    v11 = *(_QWORD *)(v8 + 16);
    if ( v9 )
    {
      if ( v11 != v10 )
        sub_CB5AE0(v8);
      v7 = sub_2207820(v9);
      sub_CB5980(v8, v7, v9, 1);
    }
    else
    {
      if ( v11 != v10 )
        sub_CB5AE0(v8);
      v7 = 0;
      sub_CB5980(v8, 0, 0, 0);
    }
  }
  if ( v22 != &v23 )
    _libc_free(v22, v7);
  sub_CB5840(v16);
  sub_A552A0((__int64)v24, v7);
}
