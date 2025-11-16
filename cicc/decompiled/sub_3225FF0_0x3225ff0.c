// Function: sub_3225FF0
// Address: 0x3225ff0
//
void __fastcall sub_3225FF0(__int64 a1, unsigned __int8 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  void (__fastcall *v7)(__int64, char, void **, __int64, __int64, __int64); // rax
  _QWORD *v8; // r13
  __int64 v9; // rax
  __m128i **v10; // r13
  __m128i v11; // [rsp+0h] [rbp-70h] BYREF
  __int64 v12; // [rsp+10h] [rbp-60h] BYREF
  void *v13[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v14; // [rsp+40h] [rbp-30h]

  if ( *(_BYTE *)(a1 + 120) )
    v6 = *(_QWORD *)(a1 + 104) + 80LL;
  else
    v6 = *(_QWORD *)(a1 + 112);
  v7 = **(void (__fastcall ***)(__int64, char, void **, __int64, __int64, __int64))v6;
  v14 = 264;
  LOBYTE(v13[0]) = a2;
  if ( v7 == sub_3225B70 )
  {
    v8 = *(_QWORD **)(v6 + 8);
    v9 = v8[1];
    if ( (unsigned __int64)(v9 + 1) > v8[2] )
    {
      sub_C8D290(*(_QWORD *)(v6 + 8), v8 + 3, v9 + 1, 1u, a5, a6);
      v9 = v8[1];
    }
    *(_BYTE *)(*v8 + v9) = a2;
    ++v8[1];
    if ( *(_BYTE *)(v6 + 24) )
    {
      v10 = *(__m128i ***)(v6 + 16);
      sub_CA0F50(v11.m128i_i64, v13);
      sub_3225850(v10, &v11);
      if ( (__int64 *)v11.m128i_i64[0] != &v12 )
        j_j___libc_free_0(v11.m128i_u64[0]);
    }
  }
  else
  {
    ((void (__fastcall *)(__int64, _QWORD, void **))v7)(v6, a2, v13);
  }
}
