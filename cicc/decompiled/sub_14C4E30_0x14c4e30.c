// Function: sub_14C4E30
// Address: 0x14c4e30
//
__int64 __fastcall sub_14C4E30(__int64 a1, unsigned int a2, int a3, int a4)
{
  int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rsi
  _BYTE *v12; // rdi
  __int64 v13; // r12
  __int64 v15; // [rsp+8h] [rbp-D8h]
  _BYTE *v16; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v17; // [rsp+28h] [rbp-B8h]
  _BYTE v18[176]; // [rsp+30h] [rbp-B0h] BYREF

  v16 = v18;
  v17 = 0x1000000000LL;
  if ( a4 )
  {
    v7 = 0;
    do
    {
      v8 = sub_1643350(*(_QWORD *)(a1 + 24));
      v9 = sub_159C470(v8, a2, 0);
      v10 = (unsigned int)v17;
      if ( (unsigned int)v17 >= HIDWORD(v17) )
      {
        v15 = v9;
        sub_16CD150(&v16, v18, 0, 8);
        v10 = (unsigned int)v17;
        v9 = v15;
      }
      ++v7;
      a2 += a3;
      *(_QWORD *)&v16[8 * v10] = v9;
      v11 = (unsigned int)(v17 + 1);
      LODWORD(v17) = v17 + 1;
    }
    while ( a4 != v7 );
    v12 = v16;
  }
  else
  {
    v12 = v18;
    v11 = 0;
  }
  v13 = sub_15A01B0(v12, v11);
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
  return v13;
}
