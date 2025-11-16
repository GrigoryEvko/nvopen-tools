// Function: sub_14C4F60
// Address: 0x14c4f60
//
__int64 __fastcall sub_14C4F60(__int64 a1, unsigned int a2, int a3, int a4)
{
  int v4; // r14d
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rsi
  int v13; // ebx
  __int64 v14; // r13
  __int64 v16; // [rsp+0h] [rbp-D0h]
  _BYTE *v17; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-B8h]
  _BYTE v19[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = a3 + a2;
  v6 = a2;
  v18 = 0x1000000000LL;
  v17 = v19;
  if ( a3 )
  {
    do
    {
      v7 = sub_1643350(*(_QWORD *)(a1 + 24));
      v8 = sub_159C470(v7, v6, 0);
      v9 = (unsigned int)v18;
      if ( (unsigned int)v18 >= HIDWORD(v18) )
      {
        v16 = v8;
        sub_16CD150(&v17, v19, 0, 8);
        v9 = (unsigned int)v18;
        v8 = v16;
      }
      ++v6;
      *(_QWORD *)&v17[8 * v9] = v8;
      LODWORD(v18) = v18 + 1;
    }
    while ( v4 != v6 );
  }
  v10 = sub_1643350(*(_QWORD *)(a1 + 24));
  v11 = sub_1599EF0(v10);
  if ( a4 )
  {
    v12 = (unsigned int)v18;
    v13 = 0;
    do
    {
      if ( HIDWORD(v18) <= (unsigned int)v12 )
      {
        sub_16CD150(&v17, v19, 0, 8);
        v12 = (unsigned int)v18;
      }
      ++v13;
      *(_QWORD *)&v17[8 * v12] = v11;
      v12 = (unsigned int)(v18 + 1);
      LODWORD(v18) = v18 + 1;
    }
    while ( a4 != v13 );
  }
  else
  {
    v12 = (unsigned int)v18;
  }
  v14 = sub_15A01B0(v17, v12);
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v17);
  return v14;
}
