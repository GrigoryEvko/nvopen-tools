// Function: sub_AC3380
// Address: 0xac3380
//
__int64 __fastcall sub_AC3380(__int64 a1, __int64 a2, unsigned int a3, unsigned __int8 a4)
{
  _QWORD *v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 i; // rax
  __int64 v11; // r12
  __int64 v13; // [rsp+8h] [rbp-C8h]
  _QWORD *v14; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v15; // [rsp+18h] [rbp-B8h]
  _QWORD v16[22]; // [rsp+20h] [rbp-B0h] BYREF

  v7 = v16;
  v14 = v16;
  v15 = 0x1000000000LL;
  if ( a3 )
  {
    v8 = v16;
    if ( a3 > 0x10uLL )
    {
      v13 = a3;
      sub_C8D5F0(&v14, v16, a3, 8);
      v7 = v14;
      v8 = &v14[(unsigned int)v15];
      v9 = &v14[v13];
      if ( v9 != v8 )
        goto LABEL_4;
    }
    else
    {
      v9 = &v16[a3];
      if ( v9 != v16 )
      {
        do
        {
LABEL_4:
          if ( v8 )
            *v8 = 0;
          ++v8;
        }
        while ( v9 != v8 );
        v7 = v14;
      }
    }
    LODWORD(v15) = a3;
  }
  if ( a3 )
  {
    for ( i = 0; ; ++i )
    {
      v7[i] = *(_QWORD *)(*(_QWORD *)(a2 + i * 8) + 8LL);
      v7 = v14;
      if ( a3 - 1 == i )
        break;
    }
  }
  v11 = sub_BD0B90(a1, v7, (unsigned int)v15, a4);
  if ( v14 != v16 )
    _libc_free(v14, v7);
  return v11;
}
