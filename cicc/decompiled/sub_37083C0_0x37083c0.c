// Function: sub_37083C0
// Address: 0x37083c0
//
__int64 __fastcall sub_37083C0(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  const void *v5; // rsi
  size_t v6; // rdx
  unsigned int v8; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v9; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v10; // [rsp+18h] [rbp-38h]

  v8 = 0;
  v2 = sub_3707C10(a1);
  sub_3704240(&v9, a2, v2);
  v3 = v9;
  v4 = v10;
  if ( v9 != v10 )
  {
    do
    {
      v5 = *(const void **)v3;
      v6 = *(_QWORD *)(v3 + 8);
      v3 += 16LL;
      v8 = sub_3707F80(a1, v5, v6);
    }
    while ( v4 != v3 );
    v4 = v9;
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  return v8;
}
