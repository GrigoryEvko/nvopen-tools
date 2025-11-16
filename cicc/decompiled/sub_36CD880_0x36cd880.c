// Function: sub_36CD880
// Address: 0x36cd880
//
__int64 __fastcall sub_36CD880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 i; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // rbx
  unsigned int v12; // r13d
  _BYTE *v13; // r14
  __int64 v14; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  _BYTE *v18; // [rsp+0h] [rbp-70h] BYREF
  __int64 v19; // [rsp+8h] [rbp-68h]
  _BYTE v20[96]; // [rsp+10h] [rbp-60h] BYREF

  v6 = a2 + 72;
  v7 = *(_QWORD *)(a2 + 80);
  v18 = v20;
  v19 = 0x600000000LL;
  if ( a2 + 72 == v7 )
  {
    i = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v7 + 32);
      if ( i != v7 + 24 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 == v7 )
        break;
      if ( !v7 )
        BUG();
    }
  }
  while ( v7 != v6 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 66 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)(i - 88) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
        v9 = **(_QWORD **)(v9 + 16);
      if ( *(_DWORD *)(v9 + 8) >> 8 == 5 )
      {
        v16 = (unsigned int)v19;
        v17 = (unsigned int)v19 + 1LL;
        if ( v17 > HIDWORD(v19) )
        {
          sub_C8D5F0((__int64)&v18, v20, v17, 8u, a5, a6);
          v16 = (unsigned int)v19;
        }
        *(_QWORD *)&v18[8 * v16] = i - 24;
        LODWORD(v19) = v19 + 1;
      }
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v7 + 32) )
    {
      v10 = v7 - 24;
      if ( !v7 )
        v10 = 0;
      if ( i != v10 + 48 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 == v7 )
        break;
      if ( !v7 )
        BUG();
    }
  }
  v11 = v18;
  v12 = 0;
  v13 = &v18[8 * (unsigned int)v19];
  if ( v13 != v18 )
  {
    do
    {
      v14 = *(_QWORD *)v11;
      v11 += 8;
      v12 |= sub_2A2DB90(v14);
    }
    while ( v13 != v11 );
    v11 = v18;
  }
  if ( v11 != v20 )
    _libc_free((unsigned __int64)v11);
  return v12;
}
