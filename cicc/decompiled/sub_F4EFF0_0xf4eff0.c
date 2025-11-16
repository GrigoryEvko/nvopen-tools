// Function: sub_F4EFF0
// Address: 0xf4eff0
//
__int64 __fastcall sub_F4EFF0(__int64 a1, __int64 a2, _BYTE *a3)
{
  _BYTE *v4; // rsi
  _BYTE *v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // rdx
  __int64 *v9; // rdi
  unsigned int v10; // r12d
  __int64 *v11; // r15
  __int64 *v12; // r13
  __int64 v13; // r14
  _BYTE *v15; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+18h] [rbp-68h]
  _BYTE v17[16]; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v18; // [rsp+30h] [rbp-50h] BYREF
  __int64 v19; // [rsp+38h] [rbp-48h]
  _BYTE v20[64]; // [rsp+40h] [rbp-40h] BYREF

  v4 = a3;
  v15 = v17;
  v16 = 0x100000000LL;
  v19 = 0x100000000LL;
  v18 = (__int64 *)v20;
  sub_AE7A40((__int64)&v15, a3, (__int64)&v18);
  v6 = v15;
  v7 = &v15[8 * (unsigned int)v16];
  if ( v15 == v7 )
  {
LABEL_7:
    v9 = v18;
    v11 = &v18[(unsigned int)v19];
    if ( v11 == v18 )
    {
      v10 = 0;
    }
    else
    {
      v12 = v18;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( a1 == sub_B12000(*v12 + 72) )
            break;
          if ( v11 == ++v12 )
            goto LABEL_13;
        }
        if ( a2 == sub_B11F60(v13 + 80) )
          goto LABEL_6;
        ++v12;
      }
      while ( v11 != v12 );
LABEL_13:
      v9 = v18;
      v10 = 0;
    }
  }
  else
  {
    while ( 1 )
    {
      v4 = *(_BYTE **)v6;
      v8 = *(_DWORD *)(*(_QWORD *)v6 + 4LL) & 0x7FFFFFF;
      if ( a1 == *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 32 * (1 - v8)) + 24LL)
        && a2 == *(_QWORD *)(*(_QWORD *)&v4[32 * (2 - v8)] + 24LL) )
      {
        break;
      }
      v6 += 8;
      if ( v7 == v6 )
        goto LABEL_7;
    }
LABEL_6:
    v9 = v18;
    v10 = 1;
  }
  if ( v9 != (__int64 *)v20 )
    _libc_free(v9, v4);
  if ( v15 != v17 )
    _libc_free(v15, v4);
  return v10;
}
