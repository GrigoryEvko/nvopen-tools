// Function: sub_15E09A0
// Address: 0x15e09a0
//
__int64 __fastcall sub_15E09A0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 result; // rax
  _BYTE v7[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v8; // [rsp+10h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 88);
  v3 = 40LL * *(_QWORD *)(a1 + 96);
  v4 = v2 + v3;
  if ( v2 != v2 + v3 )
  {
    do
    {
      v8 = 257;
      sub_164B780(v2, v7);
      v5 = v2;
      v2 += 40;
      sub_164BE60(v5);
    }
    while ( v4 != v2 );
    v4 = *(_QWORD *)(a1 + 88);
    v3 = 40LL * *(_QWORD *)(a1 + 96);
  }
  result = j_j___libc_free_0(v4, v3);
  *(_QWORD *)(a1 + 88) = 0;
  return result;
}
