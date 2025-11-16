// Function: sub_1556C90
// Address: 0x1556c90
//
__int64 __fastcall sub_1556C90(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _BYTE *v5; // r12
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // r8
  __int64 v14; // rbx
  __int64 v15; // r12
  unsigned __int64 v16; // rdi
  __int64 v17; // r14
  void *v18; // [rsp+0h] [rbp-420h] BYREF
  __int64 v19; // [rsp+8h] [rbp-418h]
  __int64 v20; // [rsp+10h] [rbp-410h]
  __int64 v21; // [rsp+18h] [rbp-408h]
  int v22; // [rsp+20h] [rbp-400h]
  __int64 v23; // [rsp+28h] [rbp-3F8h]
  _BYTE v24[40]; // [rsp+40h] [rbp-3E0h] BYREF
  __int64 v25; // [rsp+68h] [rbp-3B8h]
  __int64 v26; // [rsp+90h] [rbp-390h]
  __int64 v27; // [rsp+B8h] [rbp-368h]
  __int64 v28; // [rsp+E0h] [rbp-340h]
  unsigned __int64 v29; // [rsp+100h] [rbp-320h]
  unsigned int v30; // [rsp+108h] [rbp-318h]
  int v31; // [rsp+10Ch] [rbp-314h]
  __int64 v32; // [rsp+130h] [rbp-2F0h]
  char v33; // [rsp+150h] [rbp-2D0h]
  __int64 v34[88]; // [rsp+160h] [rbp-2C0h] BYREF

  v33 = 0;
  v5 = (_BYTE *)sub_154BC70(a3);
  if ( !v5 )
  {
    v17 = *(_QWORD *)(a1 + 48);
    v5 = v24;
    v33 = 1;
    sub_154BB30((__int64)v24, v17, 0);
  }
  sub_154B550((__int64)&v18, a2);
  sub_1556670((__int64)v34, (__int64)&v18, (__int64)v5, *(_QWORD *)(a1 + 48), 0, a4, 0);
  sub_154F560(v34, a1);
  sub_1549650(v34);
  v18 = &unk_49EF340;
  if ( v21 != v19 )
    sub_16E7BA0(&v18);
  v6 = v23;
  if ( v23 )
  {
    if ( !v22 || v19 )
    {
      v7 = v20 - v19;
    }
    else
    {
      v12 = sub_16E7720(&v18);
      v6 = v23;
      v7 = v12;
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
  result = sub_16E7960(&v18);
  if ( v33 )
  {
    j___libc_free_0(v32);
    if ( v31 )
    {
      v13 = v29;
      if ( v30 )
      {
        v14 = 8LL * v30;
        v15 = 0;
        do
        {
          v16 = *(_QWORD *)(v13 + v15);
          if ( v16 != -8 )
          {
            if ( v16 )
            {
              _libc_free(v16);
              v13 = v29;
            }
          }
          v15 += 8;
        }
        while ( v14 != v15 );
      }
    }
    else
    {
      v13 = v29;
    }
    _libc_free(v13);
    j___libc_free_0(v28);
    j___libc_free_0(v27);
    j___libc_free_0(v26);
    return j___libc_free_0(v25);
  }
  return result;
}
