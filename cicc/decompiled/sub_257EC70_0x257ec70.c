// Function: sub_257EC70
// Address: 0x257ec70
//
__int64 __fastcall sub_257EC70(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r14
  _BYTE *v4; // rdi
  __int64 v5; // r13
  unsigned __int64 *v6; // r13
  __m128i v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // rdx
  unsigned __int64 *v11; // [rsp+8h] [rbp-98h]
  char v12; // [rsp+1Eh] [rbp-82h] BYREF
  char v13; // [rsp+1Fh] [rbp-81h] BYREF
  __m128i v14; // [rsp+20h] [rbp-80h] BYREF
  _QWORD *v15; // [rsp+30h] [rbp-70h] BYREF
  __int64 v16; // [rsp+38h] [rbp-68h]
  _BYTE v17[96]; // [rsp+40h] [rbp-60h] BYREF

  v12 = 0;
  v3 = sub_250D070((_QWORD *)(a1 + 72));
  v15 = v17;
  v16 = 0x300000000LL;
  if ( !(unsigned __int8)sub_2526B50(a2, (const __m128i *)(a1 + 72), a1, (__int64)&v15, 3u, &v12, 1u)
    || (v4 = v15, (unsigned int)v16 == 1) && v3 == *v15 )
  {
    v14.m128i_i64[0] = sub_250D2C0(v3, 0);
    v14.m128i_i64[1] = v9;
    if ( *(_QWORD *)(a1 + 72) == v14.m128i_i64[0] && v9 == *(_QWORD *)(a1 + 80)
      || !(unsigned __int8)sub_257DDD0(a2, a1, &v14, 0, &v13, 0, 0) )
    {
      v8 = 0;
      *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    }
    else
    {
      v8 = 1;
    }
    v4 = v15;
  }
  else
  {
    v5 = 2LL * (unsigned int)v16;
    v11 = &v15[v5];
    if ( &v15[v5] == v15 )
    {
      v8 = 1;
    }
    else
    {
      v6 = v15;
      do
      {
        v7.m128i_i64[0] = sub_250D2C0(*v6, 0);
        v14 = v7;
        if ( !(unsigned __int8)sub_257DDD0(a2, a1, &v14, 0, &v13, 0, 0) )
        {
          v4 = v15;
          v8 = 0;
          *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
          goto LABEL_13;
        }
        v6 += 2;
      }
      while ( v11 != v6 );
      v4 = v15;
      v8 = 1;
    }
  }
LABEL_13:
  if ( v4 != v17 )
    _libc_free((unsigned __int64)v4);
  return v8;
}
