// Function: sub_26F7D90
// Address: 0x26f7d90
//
__int64 __fastcall sub_26F7D90(__int64 a1)
{
  _QWORD *v1; // rdi
  _QWORD *v2; // rax
  __int64 v3; // rbx
  unsigned int v4; // eax
  unsigned __int64 *v5; // rbx
  unsigned int v6; // r13d
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v10; // [rsp+8h] [rbp-1E8h] BYREF
  __m128i v11; // [rsp+10h] [rbp-1E0h] BYREF
  _QWORD v12[10]; // [rsp+20h] [rbp-1D0h] BYREF
  unsigned __int64 *v13; // [rsp+70h] [rbp-180h]
  unsigned int v14; // [rsp+78h] [rbp-178h]
  char v15; // [rsp+80h] [rbp-170h] BYREF

  v1 = (_QWORD *)(a1 + 24);
  v2 = (_QWORD *)v1[1];
  if ( v2 == v1 )
  {
    return 0;
  }
  else
  {
    while ( 1 )
    {
      if ( !v2 )
        BUG();
      if ( v2 + 2 != (_QWORD *)(v2[2] & 0xFFFFFFFFFFFFFFF8LL) )
        break;
      v2 = (_QWORD *)v2[1];
      if ( v1 == v2 )
        return 0;
    }
    v3 = v2[3];
    v10 = 0;
    if ( v3 )
      v3 -= 24;
    sub_B157E0((__int64)&v11, &v10);
    sub_B17430((__int64)v12, (__int64)"wholeprogramdevirt", (__int64)byte_3F871B3, 0, &v11, v3);
    if ( v10 )
      sub_B91220((__int64)&v10, v10);
    v4 = sub_B14BE0((__int64)v12);
    v5 = v13;
    v6 = v4;
    v12[0] = &unk_49D9D40;
    v7 = &v13[10 * v14];
    if ( v13 != v7 )
    {
      do
      {
        v7 -= 10;
        v8 = v7[4];
        if ( (unsigned __int64 *)v8 != v7 + 6 )
          j_j___libc_free_0(v8);
        if ( (unsigned __int64 *)*v7 != v7 + 2 )
          j_j___libc_free_0(*v7);
      }
      while ( v5 != v7 );
      v7 = v13;
    }
    if ( v7 != (unsigned __int64 *)&v15 )
      _libc_free((unsigned __int64)v7);
  }
  return v6;
}
