// Function: sub_155BB10
// Address: 0x155bb10
//
__int64 __fastcall sub_155BB10(__int64 a1, __int64 a2, __int64 a3, char a4, char a5, __m128i a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // r8
  __int64 v17; // r12
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  __int64 v21; // rax
  void *v24; // [rsp+10h] [rbp-410h] BYREF
  __int64 v25; // [rsp+18h] [rbp-408h]
  __int64 v26; // [rsp+20h] [rbp-400h]
  __int64 v27; // [rsp+28h] [rbp-3F8h]
  int v28; // [rsp+30h] [rbp-3F0h]
  __int64 v29; // [rsp+38h] [rbp-3E8h]
  _BYTE v30[40]; // [rsp+50h] [rbp-3D0h] BYREF
  __int64 v31; // [rsp+78h] [rbp-3A8h]
  __int64 v32; // [rsp+A0h] [rbp-380h]
  __int64 v33; // [rsp+C8h] [rbp-358h]
  __int64 v34; // [rsp+F0h] [rbp-330h]
  unsigned __int64 v35; // [rsp+110h] [rbp-310h]
  unsigned int v36; // [rsp+118h] [rbp-308h]
  int v37; // [rsp+11Ch] [rbp-304h]
  __int64 v38; // [rsp+140h] [rbp-2E0h]
  _QWORD v39[88]; // [rsp+160h] [rbp-2C0h] BYREF

  sub_154BB30((__int64)v30, a1, 0);
  sub_154B550((__int64)&v24, a2);
  sub_1556670((__int64)v39, (__int64)&v24, (__int64)v30, a1, a3, a5, a4);
  sub_155A0B0(v39, a1, v7, v8, v9, v10, a6);
  sub_1549650(v39);
  v24 = &unk_49EF340;
  if ( v27 != v25 )
    sub_16E7BA0(&v24);
  v11 = v29;
  if ( v29 )
  {
    if ( !v28 || v25 )
    {
      v12 = v26 - v25;
    }
    else
    {
      v21 = sub_16E7720(&v24);
      v11 = v29;
      v12 = v21;
    }
    v13 = *(_QWORD *)(v11 + 24);
    v14 = *(_QWORD *)(v11 + 8);
    if ( v12 )
    {
      if ( v13 != v14 )
        sub_16E7BA0(v11);
      v15 = sub_2207820(v12);
      sub_16E7A40(v11, v15, v12, 1);
    }
    else
    {
      if ( v13 != v14 )
        sub_16E7BA0(v11);
      sub_16E7A40(v11, 0, 0, 0);
    }
  }
  sub_16E7960(&v24);
  j___libc_free_0(v38);
  if ( v37 )
  {
    v16 = v35;
    if ( v36 )
    {
      v17 = 8LL * v36;
      v18 = 0;
      do
      {
        v19 = *(_QWORD *)(v16 + v18);
        if ( v19 != -8 && v19 )
        {
          _libc_free(v19);
          v16 = v35;
        }
        v18 += 8;
      }
      while ( v17 != v18 );
    }
  }
  else
  {
    v16 = v35;
  }
  _libc_free(v16);
  j___libc_free_0(v34);
  j___libc_free_0(v33);
  j___libc_free_0(v32);
  return j___libc_free_0(v31);
}
