// Function: sub_F0C740
// Address: 0xf0c740
//
bool __fastcall sub_F0C740(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int8 *v5; // rdi
  unsigned __int8 *v6; // rsi
  __int64 v8[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( (unsigned int)a2 <= 0x20 )
  {
    v3 = 0x100010100LL;
    if ( _bittest64(&v3, a2) )
      return 1;
  }
  v8[1] = v2;
  v4 = *(_QWORD *)(a1 + 88);
  v8[0] = (unsigned int)a2;
  v5 = *(unsigned __int8 **)(v4 + 32);
  v6 = &v5[*(_QWORD *)(v4 + 40)];
  return v6 != sub_F06B50(v5, (__int64)v6, v8);
}
