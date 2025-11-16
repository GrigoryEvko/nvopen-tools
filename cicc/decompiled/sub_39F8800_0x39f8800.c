// Function: sub_39F8800
// Address: 0x39f8800
//
__int64 __fastcall sub_39F8800(_QWORD *a1, int a2, __int64 a3, int a4, int a5, int a6, char a7)
{
  __int64 v7; // rax
  __m128i *v10; // rdi
  __m128i *v11; // rsi
  __int64 i; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  char v15; // [rsp-8h] [rbp-230h]
  __int64 v16; // [rsp+0h] [rbp-228h] BYREF
  __m128i v17[15]; // [rsp+8h] [rbp-220h] BYREF
  __m128i v18[9]; // [rsp+F8h] [rbp-130h] BYREF
  __int64 v19; // [rsp+190h] [rbp-98h]
  __int64 v20; // [rsp+1F0h] [rbp-38h]
  __int64 v21; // [rsp+1F8h] [rbp-30h]
  __int64 retaddr; // [rsp+230h] [rbp+8h]

  v21 = a3;
  v20 = v7;
  if ( !a1[2] )
    return sub_39F8140(a1, a2, a3, a4, a5, a6, v15);
  sub_39F7A80(v17, (__int64)&a7, retaddr);
  v10 = v18;
  v11 = v17;
  for ( i = 60; i; --i )
  {
    v10->m128i_i32[0] = v11->m128i_i32[0];
    v11 = (__m128i *)((char *)v11 + 4);
    v10 = (__m128i *)((char *)v10 + 4);
  }
  if ( (unsigned int)sub_39F7D50(a1, v18, &v16) != 7 )
    abort();
  sub_39F5CF0((__int64)v17, (__int64)v18);
  v13 = v19;
  nullsub_2004();
  *(__int64 *)((char *)&retaddr + v14) = v13;
  return v20;
}
