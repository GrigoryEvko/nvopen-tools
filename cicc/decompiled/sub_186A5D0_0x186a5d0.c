// Function: sub_186A5D0
// Address: 0x186a5d0
//
__int64 __fastcall sub_186A5D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rsi
  unsigned int v4; // r13d
  char *v5; // rbx
  char *v6; // r12
  char *v7; // rdi
  __int64 v9; // rax
  const char *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // [rsp+8h] [rbp-238h] BYREF
  __m128i v14[2]; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v15[11]; // [rsp+30h] [rbp-210h] BYREF
  char *v16; // [rsp+88h] [rbp-1B8h]
  unsigned int v17; // [rsp+90h] [rbp-1B0h]
  char v18; // [rsp+98h] [rbp-1A8h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v13 = 0;
  if ( v2 )
    v2 -= 24;
  sub_15C9090((__int64)v14, &v13);
  sub_15CA330((__int64)v15, (__int64)"inline", (__int64)byte_3F871B3, 0, v14, v2);
  v3 = v13;
  if ( v13 )
    sub_161E7C0((__int64)&v13, v13);
  if ( (unsigned __int8)sub_15C8000((__int64)v15, v3) && sub_1626D20(a1) )
  {
    v4 = 0;
    if ( sub_1626D20(a1) )
    {
      v9 = sub_1626D20(a1);
      if ( *(_BYTE *)v9 == 15 || (v9 = *(_QWORD *)(v9 - 8LL * *(unsigned int *)(v9 + 8))) != 0 )
      {
        v10 = *(const char **)(v9 - 8LL * *(unsigned int *)(v9 + 8));
        if ( v10 )
        {
          v10 = (const char *)sub_161E970((__int64)v10);
          v12 = v11;
        }
        else
        {
          v12 = 0;
        }
      }
      else
      {
        v12 = 0;
        v10 = byte_3F871B3;
      }
      v4 = sub_1C31440(v10, v12);
    }
  }
  else
  {
    v4 = 1;
  }
  v5 = v16;
  v15[0] = &unk_49ECF68;
  v6 = &v16[88 * v17];
  if ( v16 != v6 )
  {
    do
    {
      v6 -= 88;
      v7 = (char *)*((_QWORD *)v6 + 4);
      if ( v7 != v6 + 48 )
        j_j___libc_free_0(v7, *((_QWORD *)v6 + 6) + 1LL);
      if ( *(char **)v6 != v6 + 16 )
        j_j___libc_free_0(*(_QWORD *)v6, *((_QWORD *)v6 + 2) + 1LL);
    }
    while ( v5 != v6 );
    v6 = v16;
  }
  if ( v6 != &v18 )
    _libc_free((unsigned __int64)v6);
  return v4;
}
