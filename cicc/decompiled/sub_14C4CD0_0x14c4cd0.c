// Function: sub_14C4CD0
// Address: 0x14c4cd0
//
__int64 __fastcall sub_14C4CD0(__int64 a1, int a2, int a3)
{
  unsigned int v4; // ebx
  int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v13; // [rsp+10h] [rbp-D0h]
  unsigned int i; // [rsp+1Ch] [rbp-C4h]
  _BYTE *v15; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+28h] [rbp-B8h]
  _BYTE v17[176]; // [rsp+30h] [rbp-B0h] BYREF

  v15 = v17;
  v16 = 0x1000000000LL;
  if ( a2 )
  {
    for ( i = 0; i != a2; ++i )
    {
      v4 = i;
      v5 = 0;
      if ( a3 )
      {
        do
        {
          v6 = sub_1643350(*(_QWORD *)(a1 + 24));
          v7 = sub_159C470(v6, v4, 0);
          v8 = (unsigned int)v16;
          if ( (unsigned int)v16 >= HIDWORD(v16) )
          {
            v13 = v7;
            sub_16CD150(&v15, v17, 0, 8);
            v8 = (unsigned int)v16;
            v7 = v13;
          }
          ++v5;
          v4 += a2;
          *(_QWORD *)&v15[8 * v8] = v7;
          LODWORD(v16) = v16 + 1;
        }
        while ( a3 != v5 );
      }
    }
    v9 = v15;
    v10 = (unsigned int)v16;
  }
  else
  {
    v9 = v17;
    v10 = 0;
  }
  v11 = sub_15A01B0(v9, v10);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  return v11;
}
