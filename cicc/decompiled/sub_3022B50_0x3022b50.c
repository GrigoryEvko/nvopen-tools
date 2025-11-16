// Function: sub_3022B50
// Address: 0x3022b50
//
void __fastcall sub_3022B50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  _BYTE *v5; // rax
  void *v6; // rdx
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __m128i *dest; // [rsp+0h] [rbp-80h]
  __m128i v10; // [rsp+10h] [rbp-70h] BYREF
  __int64 v11; // [rsp+20h] [rbp-60h]
  void *src[2]; // [rsp+30h] [rbp-50h] BYREF
  __m128i v13; // [rsp+40h] [rbp-40h] BYREF
  char v14; // [rsp+50h] [rbp-30h]

  v11 = 0;
  dest = 0;
  v10 = 0;
  sub_314D260(src, a2, 1);
  if ( v14 )
  {
    dest = &v10;
    if ( src[0] == &v13 )
    {
      v10 = _mm_load_si128(&v13);
    }
    else
    {
      dest = (__m128i *)src[0];
      v10.m128i_i64[0] = v13.m128i_i64[0];
    }
    LOBYTE(v11) = 1;
    v4 = sub_CB6200(a3, (unsigned __int8 *)dest, (size_t)src[1]);
    v5 = *(_BYTE **)(v4 + 32);
    if ( *(_BYTE **)(v4 + 24) == v5 )
    {
      sub_CB6200(v4, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v5 = 10;
      ++*(_QWORD *)(v4 + 32);
    }
  }
  src[0] = (void *)sub_CE9340(a2);
  if ( BYTE4(src[0]) )
  {
    v6 = *(void **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v6 <= 0xEu )
    {
      a3 = sub_CB6200(a3, ".local_maxnreg ", 0xFu);
    }
    else
    {
      qmemcpy(v6, ".local_maxnreg ", 15);
      *(_QWORD *)(a3 + 32) += 15LL;
    }
    v7 = sub_CB59D0(a3, LODWORD(src[0]));
    v8 = *(_BYTE **)(v7 + 32);
    if ( *(_BYTE **)(v7 + 24) == v8 )
    {
      sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v8 = 10;
      ++*(_QWORD *)(v7 + 32);
    }
  }
  if ( (_BYTE)v11 )
  {
    LOBYTE(v11) = 0;
    if ( dest != &v10 )
      j_j___libc_free_0((unsigned __int64)dest);
  }
}
