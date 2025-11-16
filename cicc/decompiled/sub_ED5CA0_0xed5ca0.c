// Function: sub_ED5CA0
// Address: 0xed5ca0
//
unsigned __int64 *__fastcall sub_ED5CA0(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r13
  char v7; // r12
  __int64 v8; // rbx
  char v9; // r10
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  char v14; // [rsp+1Fh] [rbp-A1h]
  __int64 v15; // [rsp+38h] [rbp-88h]
  __int64 v16; // [rsp+48h] [rbp-78h] BYREF
  unsigned __int64 v17[2]; // [rsp+50h] [rbp-70h] BYREF
  char v18; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v19; // [rsp+70h] [rbp-50h] BYREF
  size_t v20; // [rsp+78h] [rbp-48h]
  _QWORD v21[8]; // [rsp+80h] [rbp-40h] BYREF

  v6 = a2;
  v7 = a5;
  v8 = *(_QWORD *)(a3 + 32);
  v14 = a4;
  v15 = a3 + 24;
  if ( v8 == a3 + 24 )
  {
LABEL_10:
    v17[0] = (unsigned __int64)&v18;
    v17[1] = 0x200000000LL;
    v11 = *(_QWORD *)(a3 + 16);
    if ( a3 + 8 == v11 )
    {
LABEL_24:
      *(_BYTE *)(v6 + 392) = 0;
      sub_ED4D20(v6, a2, a3, a4, a5, a6);
      *a1 = 1;
    }
    else
    {
      while ( 1 )
      {
        if ( !v11 )
          BUG();
        if ( (*(_BYTE *)(v11 - 49) & 0x30) == 0x30 )
        {
          a2 = 19;
          if ( sub_B91C10(v11 - 56, 19) )
          {
            sub_ED15E0((__int64 *)&v19, v11 - 56, v14);
            a2 = v6;
            sub_ED5BE0((unsigned __int64 *)&v16, v6, v11 - 56, v19, v20);
            if ( v19 != v21 )
            {
              a2 = v21[0] + 1LL;
              j_j___libc_free_0(v19, v21[0] + 1LL);
            }
            v12 = v16 & 0xFFFFFFFFFFFFFFFELL;
            if ( (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              break;
          }
        }
        v11 = *(_QWORD *)(v11 + 8);
        if ( a3 + 8 == v11 )
          goto LABEL_24;
      }
      v16 = 0;
      *a1 = v12 | 1;
      sub_9C66B0(&v16);
    }
  }
  else
  {
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      if ( (*(_BYTE *)(v8 - 49) & 0x10) != 0 )
      {
        sub_ED29C0((__int64 *)&v19, v8 - 56, v14);
        sub_ED40D0(v17, v6, v8 - 56, v19, v20, v7);
        v9 = v14;
        if ( v19 != v21 )
        {
          j_j___libc_free_0(v19, v21[0] + 1LL);
          v9 = v14;
        }
        if ( (v17[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          *a1 = v17[0] & 0xFFFFFFFFFFFFFFFELL | 1;
          return a1;
        }
        sub_ED2A00((__int64 *)&v19, v8 - 56, v9);
        a2 = v6;
        sub_ED40D0(v17, v6, v8 - 56, v19, v20, v7);
        if ( v19 != v21 )
        {
          a2 = v21[0] + 1LL;
          j_j___libc_free_0(v19, v21[0] + 1LL);
        }
        if ( (v17[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          break;
      }
      v8 = *(_QWORD *)(v8 + 8);
      if ( v15 == v8 )
        goto LABEL_10;
    }
    *a1 = v17[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  return a1;
}
