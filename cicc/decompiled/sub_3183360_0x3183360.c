// Function: sub_3183360
// Address: 0x3183360
//
_QWORD *__fastcall sub_3183360(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4)
{
  _QWORD *v4; // rax
  __int64 v5; // rsi
  _QWORD *v6; // r13
  _QWORD *v7; // rbx
  _QWORD *v8; // r12
  __int64 v9; // rax
  unsigned __int64 *v10; // rax
  unsigned __int64 v11; // r14
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v14; // [rsp+8h] [rbp-38h]
  __int64 v15; // [rsp+10h] [rbp-30h]
  unsigned int v16; // [rsp+18h] [rbp-28h]

  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v4 = sub_3182B00(a1, a2, a3, a4, (__int64)&v13);
  v5 = v16;
  v6 = v4;
  if ( v16 )
  {
    v7 = v14;
    v8 = &v14[2 * v16];
    do
    {
      if ( *v7 != -4096 && *v7 != -8192 )
      {
        v9 = v7[1];
        if ( v9 )
        {
          if ( (v9 & 4) != 0 )
          {
            v10 = (unsigned __int64 *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
            v11 = (unsigned __int64)v10;
            if ( v10 )
            {
              if ( (unsigned __int64 *)*v10 != v10 + 2 )
                _libc_free(*v10);
              j_j___libc_free_0(v11);
            }
          }
        }
      }
      v7 += 2;
    }
    while ( v8 != v7 );
    v5 = v16;
  }
  sub_C7D6A0((__int64)v14, 16 * v5, 8);
  return v6;
}
