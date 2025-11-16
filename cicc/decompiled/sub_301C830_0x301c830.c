// Function: sub_301C830
// Address: 0x301c830
//
__int64 __fastcall sub_301C830(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // eax
  _BYTE *v4; // rbx
  unsigned int v5; // r13d
  _BYTE *v6; // r12
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  _QWORD *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // r14
  _BYTE v15[4]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v16; // [rsp+4h] [rbp-7Ch]
  __int64 v17; // [rsp+Ch] [rbp-74h]
  __int128 v18; // [rsp+14h] [rbp-6Ch]
  __int64 v19; // [rsp+24h] [rbp-5Ch]
  __int64 v20; // [rsp+30h] [rbp-50h]
  __int64 v21; // [rsp+38h] [rbp-48h]
  __int64 v22; // [rsp+40h] [rbp-40h]
  unsigned int v23; // [rsp+48h] [rbp-38h]
  _BYTE *v24; // [rsp+50h] [rbp-30h]
  __int64 v25; // [rsp+58h] [rbp-28h]
  _BYTE v26[32]; // [rsp+60h] [rbp-20h] BYREF

  v2 = *(_BYTE *)(a1 + 169);
  v16 = 0;
  v17 = 0;
  v15[0] = v2;
  v18 = 0u;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = v26;
  v25 = 0;
  v3 = sub_301BCE0((__int64)v15, a2);
  v4 = v24;
  v5 = v3;
  v6 = &v24[32 * (unsigned int)v25];
  if ( v24 != v6 )
  {
    do
    {
      v7 = *((_QWORD *)v6 - 3);
      v6 -= 32;
      if ( v7 )
        j_j___libc_free_0(v7);
    }
    while ( v4 != v6 );
    v6 = v24;
  }
  if ( v6 != v26 )
    _libc_free((unsigned __int64)v6);
  sub_C7D6A0(v21, 16LL * v23, 8);
  v8 = HIDWORD(v19);
  if ( HIDWORD(v19) )
  {
    v9 = *(_QWORD **)((char *)&v18 + 4);
    v10 = *(_QWORD *)((char *)&v18 + 4) + 16LL * HIDWORD(v19);
    do
    {
      if ( *v9 != -8192 && *v9 != -4096 )
      {
        v11 = v9[1];
        if ( v11 )
        {
          if ( (v11 & 4) != 0 )
          {
            v12 = (unsigned __int64 *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
            v13 = (unsigned __int64)v12;
            if ( v12 )
            {
              if ( (unsigned __int64 *)*v12 != v12 + 2 )
                _libc_free(*v12);
              j_j___libc_free_0(v13);
            }
          }
        }
      }
      v9 += 2;
    }
    while ( (_QWORD *)v10 != v9 );
    v8 = HIDWORD(v19);
  }
  sub_C7D6A0(*(__int64 *)((char *)&v18 + 4), 16 * v8, 8);
  return v5;
}
