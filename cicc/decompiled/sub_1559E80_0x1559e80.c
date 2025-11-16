// Function: sub_1559E80
// Address: 0x1559e80
//
__int64 __fastcall sub_1559E80(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r8
  __int64 v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  __int64 v16; // rax
  void *v19; // [rsp+10h] [rbp-410h] BYREF
  __int64 v20; // [rsp+18h] [rbp-408h]
  __int64 v21; // [rsp+20h] [rbp-400h]
  __int64 v22; // [rsp+28h] [rbp-3F8h]
  int v23; // [rsp+30h] [rbp-3F0h]
  __int64 v24; // [rsp+38h] [rbp-3E8h]
  _BYTE v25[40]; // [rsp+50h] [rbp-3D0h] BYREF
  __int64 v26; // [rsp+78h] [rbp-3A8h]
  __int64 v27; // [rsp+A0h] [rbp-380h]
  __int64 v28; // [rsp+C8h] [rbp-358h]
  __int64 v29; // [rsp+F0h] [rbp-330h]
  unsigned __int64 v30; // [rsp+110h] [rbp-310h]
  unsigned int v31; // [rsp+118h] [rbp-308h]
  int v32; // [rsp+11Ch] [rbp-304h]
  __int64 v33; // [rsp+140h] [rbp-2E0h]
  __int64 v34[88]; // [rsp+160h] [rbp-2C0h] BYREF

  sub_154BB30((__int64)v25, *(_QWORD *)(a1 + 40), 0);
  sub_154B550((__int64)&v19, a2);
  sub_1556670((__int64)v34, (__int64)&v19, (__int64)v25, *(_QWORD *)(a1 + 40), a3, a5, a4);
  sub_1559290(v34, a1);
  sub_1549650(v34);
  v19 = &unk_49EF340;
  if ( v22 != v20 )
    sub_16E7BA0(&v19);
  v6 = v24;
  if ( v24 )
  {
    if ( !v23 || v20 )
    {
      v7 = v21 - v20;
    }
    else
    {
      v16 = sub_16E7720(&v19);
      v6 = v24;
      v7 = v16;
    }
    v8 = *(_QWORD *)(v6 + 24);
    v9 = *(_QWORD *)(v6 + 8);
    if ( v7 )
    {
      if ( v8 != v9 )
        sub_16E7BA0(v6);
      v10 = sub_2207820(v7);
      sub_16E7A40(v6, v10, v7, 1);
    }
    else
    {
      if ( v8 != v9 )
        sub_16E7BA0(v6);
      sub_16E7A40(v6, 0, 0, 0);
    }
  }
  sub_16E7960(&v19);
  j___libc_free_0(v33);
  if ( v32 )
  {
    v11 = v30;
    if ( v31 )
    {
      v12 = 8LL * v31;
      v13 = 0;
      do
      {
        v14 = *(_QWORD *)(v11 + v13);
        if ( v14 != -8 && v14 )
        {
          _libc_free(v14);
          v11 = v30;
        }
        v13 += 8;
      }
      while ( v12 != v13 );
    }
  }
  else
  {
    v11 = v30;
  }
  _libc_free(v11);
  j___libc_free_0(v29);
  j___libc_free_0(v28);
  j___libc_free_0(v27);
  return j___libc_free_0(v26);
}
