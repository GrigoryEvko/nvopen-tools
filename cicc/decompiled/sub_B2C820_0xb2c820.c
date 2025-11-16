// Function: sub_B2C820
// Address: 0xb2c820
//
void __fastcall sub_B2C820(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r13
  __int64 v5; // r14
  _QWORD *v6; // rsi
  const void *v7; // rax
  size_t v8; // rdx
  size_t v9; // rdi
  const void *v10; // [rsp+0h] [rbp-120h]
  size_t v11; // [rsp+10h] [rbp-110h]
  size_t v12; // [rsp+10h] [rbp-110h]
  _QWORD v14[4]; // [rsp+20h] [rbp-100h] BYREF
  __int16 v15; // [rsp+40h] [rbp-E0h]
  _BYTE *v16; // [rsp+50h] [rbp-D0h] BYREF
  size_t v17; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v18; // [rsp+60h] [rbp-C0h]
  _BYTE v19[184]; // [rsp+68h] [rbp-B8h] BYREF

  if ( (*(_BYTE *)(a1 + 2) & 1) == 0 )
  {
    sub_B2C790(a1);
    *(_WORD *)(a1 + 2) |= 1u;
  }
  if ( (*(_BYTE *)(a2 + 2) & 1) == 0 )
  {
    *(_QWORD *)(a1 + 96) = *(_QWORD *)(a2 + 96);
    *(_QWORD *)(a2 + 96) = 0;
    v3 = *(_QWORD *)(a1 + 96);
    v4 = v3 + 40LL * *(_QWORD *)(a1 + 104);
    v5 = v3;
    if ( v3 == v4 )
    {
LABEL_18:
      *(_WORD *)(a1 + 2) &= ~1u;
      *(_WORD *)(a2 + 2) |= 1u;
      return;
    }
    while ( 1 )
    {
      v16 = v19;
      v17 = 0;
      v18 = 128;
      if ( (*(_BYTE *)(v5 + 7) & 0x10) != 0 )
      {
        v7 = (const void *)sub_BD5D20(v5);
        v17 = 0;
        if ( v8 > v18 )
        {
          v10 = v7;
          v12 = v8;
          sub_C8D290(&v16, v19, v8, 1);
          v8 = v12;
          v9 = v17;
          v7 = v10;
          if ( v12 )
          {
LABEL_15:
            v11 = v8;
            memcpy(&v16[v9], v7, v8);
            v9 = v17;
            v8 = v11;
          }
          v17 = v9 + v8;
          if ( v9 + v8 )
          {
            v15 = 257;
            sub_BD6B50(v5, v14);
          }
          goto LABEL_6;
        }
        if ( v8 )
        {
          v9 = 0;
          goto LABEL_15;
        }
      }
LABEL_6:
      v6 = (_QWORD *)a1;
      sub_B2BAD0(v5, a1);
      if ( v17 )
      {
        v6 = v14;
        v14[1] = v17;
        v15 = 261;
        v14[0] = v16;
        sub_BD6B50(v5, v14);
      }
      if ( v16 != v19 )
        _libc_free(v16, v6);
      v5 += 40;
      if ( v4 == v5 )
        goto LABEL_18;
    }
  }
}
