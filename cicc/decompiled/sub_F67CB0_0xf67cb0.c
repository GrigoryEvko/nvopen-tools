// Function: sub_F67CB0
// Address: 0xf67cb0
//
__int64 __fastcall sub_F67CB0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r12
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r13
  char v19; // [rsp+4h] [rbp-ACh]
  __int64 v23; // [rsp+20h] [rbp-90h]
  _BYTE *v24; // [rsp+30h] [rbp-80h] BYREF
  __int64 v25; // [rsp+38h] [rbp-78h]
  _BYTE v26[112]; // [rsp+40h] [rbp-70h] BYREF

  v6 = 0x800000000LL;
  v19 = a5;
  v7 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 16LL);
  v23 = **(_QWORD **)(a1 + 32);
  v24 = v26;
  v25 = 0x800000000LL;
  if ( v7 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v8 - 30) <= 0xAu )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        goto LABEL_21;
    }
    v9 = *(_QWORD *)(v8 + 40);
    if ( !*(_BYTE *)(a1 + 84) )
      goto LABEL_11;
LABEL_4:
    v10 = *(_QWORD **)(a1 + 64);
    v11 = &v10[*(unsigned int *)(a1 + 76)];
    if ( v10 == v11 )
    {
LABEL_12:
      v13 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v13 == v9 + 48 )
        goto LABEL_28;
      if ( !v13 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_28:
        BUG();
      if ( *(_BYTE *)(v13 - 24) == 33 )
      {
        v17 = 0;
        goto LABEL_24;
      }
      v14 = (unsigned int)v25;
      v15 = (unsigned int)v25 + 1LL;
      if ( v15 > HIDWORD(v25) )
      {
        v6 = (__int64)v26;
        sub_C8D5F0((__int64)&v24, v26, v15, 8u, a5, a6);
        v14 = (unsigned int)v25;
      }
      *(_QWORD *)&v24[8 * v14] = v9;
      LODWORD(v25) = v25 + 1;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v7 )
        goto LABEL_9;
    }
    else
    {
      while ( v9 != *v10 )
      {
        if ( v11 == ++v10 )
          goto LABEL_12;
      }
      while ( 1 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          break;
LABEL_9:
        v12 = *(_QWORD *)(v7 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v12 - 30) <= 0xAu )
        {
          v9 = *(_QWORD *)(v12 + 40);
          if ( *(_BYTE *)(a1 + 84) )
            goto LABEL_4;
LABEL_11:
          v6 = v9;
          if ( !sub_C8CA60(a1 + 56, v9) )
            goto LABEL_12;
        }
      }
    }
    v6 = (__int64)v24;
    v16 = (unsigned int)v25;
  }
  else
  {
LABEL_21:
    v6 = (__int64)v26;
    v16 = 0;
  }
  v17 = sub_F40FB0(v23, (__int64 **)v6, v16, ".preheader", a2, a3, a4, v19);
  if ( v17 )
  {
    v6 = (__int64)&v24;
    sub_F672C0(v17, (__int64)&v24, a1);
  }
LABEL_24:
  if ( v24 != v26 )
    _libc_free(v24, v6);
  return v17;
}
