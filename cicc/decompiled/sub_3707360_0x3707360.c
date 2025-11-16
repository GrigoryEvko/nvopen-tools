// Function: sub_3707360
// Address: 0x3707360
//
__int64 *__fastcall sub_3707360(__int64 *a1, _QWORD *a2, __int64 a3, __int64 *a4, int a5)
{
  _BYTE *v8; // rsi
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  volatile signed __int32 *v11; // rdi
  void **v13; // [rsp+28h] [rbp-78h] BYREF
  void *v14; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v15; // [rsp+38h] [rbp-68h]
  void *v16; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v17; // [rsp+48h] [rbp-58h] BYREF
  _BYTE *v18; // [rsp+50h] [rbp-50h]
  _BYTE *v19; // [rsp+58h] [rbp-48h]
  void **v20; // [rsp+60h] [rbp-40h] BYREF

  v15 = 0;
  v17 = 0;
  v18 = 0;
  v14 = &unk_4A3C658;
  v16 = &unk_4A35300;
  v19 = 0;
  if ( a5 )
  {
    v20 = (void **)a4;
  }
  else
  {
    v20 = &v16;
    v13 = &v14;
    sub_37071D0((__int64)&v17, 0, &v13);
    v8 = v18;
    v13 = (void **)a4;
    if ( v19 == v18 )
    {
      sub_37071D0((__int64)&v17, v18, &v13);
    }
    else
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = a4;
        v8 = v18;
      }
      v18 = v8 + 8;
    }
  }
  sub_3706910(a1, (__int64 **)&v20, a2);
  v16 = &unk_4A35300;
  if ( v17 )
    j_j___libc_free_0(v17);
  v9 = v15;
  v14 = &unk_4A3C658;
  if ( v15 )
  {
    v10 = *(_QWORD *)(v15 + 112);
    if ( v10 != v15 + 128 )
      _libc_free(v10);
    v11 = *(volatile signed __int32 **)(v9 + 48);
    *(_QWORD *)(v9 + 32) = &unk_49E6870;
    if ( v11 )
      sub_A191D0(v11);
    j_j___libc_free_0(v9);
  }
  return a1;
}
