// Function: sub_2557AA0
// Address: 0x2557aa0
//
__m128i *__fastcall sub_2557AA0(__m128i *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  int v6; // eax
  const char *v7; // rsi
  __int64 v9[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v10; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v11; // [rsp+30h] [rbp-90h] BYREF
  __int64 v12; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v13; // [rsp+50h] [rbp-70h] BYREF
  int v14; // [rsp+58h] [rbp-68h]
  _QWORD v15[2]; // [rsp+60h] [rbp-60h] BYREF
  __m128i v16; // [rsp+70h] [rbp-50h] BYREF
  __int64 v17; // [rsp+80h] [rbp-40h] BYREF

  v2 = a2;
  v3 = *(unsigned int *)(a2 + 256);
  if ( v3 <= 9 )
  {
    a2 = 1;
  }
  else if ( v3 <= 0x63 )
  {
    a2 = 2;
  }
  else if ( v3 <= 0x3E7 )
  {
    a2 = 3;
  }
  else if ( v3 <= 0x270F )
  {
    a2 = 4;
  }
  else
  {
    v4 = *(unsigned int *)(a2 + 256);
    LODWORD(a2) = 1;
    while ( 1 )
    {
      v5 = v4;
      v6 = a2;
      a2 = (unsigned int)(a2 + 4);
      v4 /= 0x2710u;
      if ( v5 <= 0x1869F )
        break;
      if ( v5 <= 0xF423F )
      {
        a2 = (unsigned int)(v6 + 5);
        break;
      }
      if ( v5 <= (unsigned __int64)&loc_98967F )
      {
        a2 = (unsigned int)(v6 + 6);
        break;
      }
      if ( v5 <= 0x5F5E0FF )
      {
        a2 = (unsigned int)(v6 + 7);
        break;
      }
    }
  }
  v13 = v15;
  sub_2240A50((__int64 *)&v13, a2, 0);
  sub_1249540(v13, v14, v3);
  v7 = "eliminate";
  if ( !*(_BYTE *)(v2 + 296) )
    v7 = "specialize";
  sub_253C590(v9, v7);
  sub_94F930(&v11, (__int64)v9, " indirect call site with ");
  sub_8FD5D0(&v16, (__int64)&v11, &v13);
  sub_94F930(a1, (__int64)&v16, " functions");
  if ( (__int64 *)v16.m128i_i64[0] != &v17 )
    j_j___libc_free_0(v16.m128i_u64[0]);
  if ( (__int64 *)v11.m128i_i64[0] != &v12 )
    j_j___libc_free_0(v11.m128i_u64[0]);
  if ( (__int64 *)v9[0] != &v10 )
    j_j___libc_free_0(v9[0]);
  if ( v13 != (_BYTE *)v15 )
    j_j___libc_free_0((unsigned __int64)v13);
  return a1;
}
