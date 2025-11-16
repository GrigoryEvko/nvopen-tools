// Function: sub_7433F0
// Address: 0x7433f0
//
__int64 __fastcall sub_7433F0(__m128i *a1, __m128i *a2, __int64 a3, _DWORD *a4, int a5, int *a6, __int64 *a7)
{
  __m128i *v9; // r13
  __m128i *v11; // rax
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // r12
  const __m128i *v20; // [rsp+0h] [rbp-70h]
  _QWORD *v21; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+28h] [rbp-48h] BYREF
  __m128i *v24; // [rsp+30h] [rbp-40h] BYREF
  const __m128i *v25; // [rsp+38h] [rbp-38h] BYREF

  v9 = a1;
  v23 = 0;
  v11 = (__m128i *)sub_724DC0();
  v24 = v11;
  if ( a1 )
  {
    v12 = (unsigned __int64 *)&v23;
    while ( 1 )
    {
      v13 = sub_7410C0(v9, a2, a3, 0, a4, a5, a6, a7, v11, &v25);
      *v12 = v13;
      v14 = v13;
      if ( *a6 )
        break;
      if ( v13
        || (!v25 ? (v20 = v24, v15 = sub_73A720(v24, (__int64)a2)) : (v20 = v25, v15 = sub_730690((__int64)v25)),
            v16 = v20[8].m128i_i64[0],
            v21 = v15,
            v17 = sub_8D32E0(v16),
            v14 = (unsigned __int64)v21,
            !v17) )
      {
        *v12 = v14;
        v9 = (__m128i *)v9[1].m128i_i64[0];
        v12 = (unsigned __int64 *)(v14 + 16);
        if ( !v9 )
          break;
      }
      else
      {
        *((_BYTE *)v21 + 25) |= 1u;
        *v12 = (unsigned __int64)v21;
        v9 = (__m128i *)v9[1].m128i_i64[0];
        v12 = v21 + 2;
        if ( !v9 )
          break;
      }
      v11 = v24;
    }
    v18 = v23;
  }
  else
  {
    v18 = 0;
  }
  sub_724E30((__int64)&v24);
  return v18;
}
