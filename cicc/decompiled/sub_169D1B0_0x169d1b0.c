// Function: sub_169D1B0
// Address: 0x169d1b0
//
__int64 __fastcall sub_169D1B0(__int64 a1, unsigned int a2, char a3)
{
  _QWORD *v3; // r14
  void *v4; // rax
  void *v5; // rsi
  _QWORD *v6; // r13
  void *v7; // rax
  void *v8; // rsi
  unsigned __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-28h]

  if ( !a3 )
  {
    v11 = a2;
    if ( a2 <= 0x40 )
      v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a2;
    else
      sub_16A4EF0(&v10, -1, 1);
    v4 = sub_16982C0();
    v5 = &unk_42AE990;
    v3 = (_QWORD *)(a1 + 8);
    if ( v4 != &unk_42AE990 )
      goto LABEL_6;
    goto LABEL_17;
  }
  if ( a2 == 64 )
  {
    v11 = 64;
    v6 = (_QWORD *)(a1 + 8);
    v10 = -1;
    v7 = sub_16982C0();
    v8 = &unk_42AE9D0;
    if ( v7 != &unk_42AE9D0 )
      goto LABEL_9;
    goto LABEL_21;
  }
  if ( a2 <= 0x40 )
  {
    if ( a2 == 16 )
    {
      v11 = 16;
      v6 = (_QWORD *)(a1 + 8);
      v10 = 0xFFFF;
      v7 = sub_16982C0();
      v8 = &unk_42AE9F0;
      if ( v7 != &unk_42AE9F0 )
      {
LABEL_9:
        sub_169D050((__int64)v6, v8, (__int64 *)&v10);
        goto LABEL_10;
      }
    }
    else
    {
      v11 = 32;
      v6 = (_QWORD *)(a1 + 8);
      v10 = 0xFFFFFFFFLL;
      v7 = sub_16982C0();
      v8 = &unk_42AE9E0;
      if ( v7 != &unk_42AE9E0 )
        goto LABEL_9;
    }
LABEL_21:
    sub_169D060(v6, (__int64)v7, (__int64 *)&v10);
    goto LABEL_10;
  }
  if ( a2 == 80 )
  {
    v11 = 80;
    v3 = (_QWORD *)(a1 + 8);
    sub_16A4EF0(&v10, -1, 1);
    v4 = sub_16982C0();
    v5 = &unk_42AE9B0;
    if ( v4 != &unk_42AE9B0 )
      goto LABEL_6;
LABEL_17:
    sub_169D060(v3, (__int64)v4, (__int64 *)&v10);
    goto LABEL_10;
  }
  v11 = 128;
  v3 = (_QWORD *)(a1 + 8);
  sub_16A4EF0(&v10, -1, 1);
  v4 = sub_16982C0();
  v5 = &unk_42AE9C0;
  if ( v4 == &unk_42AE9C0 )
    goto LABEL_17;
LABEL_6:
  sub_169D050((__int64)v3, v5, (__int64 *)&v10);
LABEL_10:
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return a1;
}
