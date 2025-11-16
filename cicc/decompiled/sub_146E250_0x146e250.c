// Function: sub_146E250
// Address: 0x146e250
//
__int64 __fastcall sub_146E250(__int64 a1, __int64 a2, __int16 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 result; // rax
  _BYTE *v9; // rdi
  void *v10; // rax
  __int64 *v11; // r10
  void *v12; // r13
  size_t v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // [rsp+8h] [rbp-F8h]
  __int64 v19; // [rsp+10h] [rbp-F0h]
  __int64 v20; // [rsp+18h] [rbp-E8h]
  __int64 v21; // [rsp+18h] [rbp-E8h]
  __int64 v23; // [rsp+28h] [rbp-D8h]
  __int64 v24; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD v25[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v26[176]; // [rsp+50h] [rbp-B0h] BYREF

  v25[0] = v26;
  v25[1] = 0x2000000000LL;
  sub_16BD3E0(v25, 5);
  v4 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v4 )
  {
    v5 = 8 * v4;
    v6 = 0;
    do
    {
      v7 = *(_QWORD *)(*(_QWORD *)a2 + v6);
      v6 += 8;
      sub_16BD4C0(v25, v7);
    }
    while ( v5 != v6 );
  }
  v24 = 0;
  result = sub_16BDDE0(a1 + 816, v25, &v24);
  if ( !result )
  {
    v10 = (void *)sub_145CBF0((__int64 *)(a1 + 864), 8LL * *(unsigned int *)(a2 + 8), 8);
    v11 = (__int64 *)(a1 + 864);
    v12 = v10;
    v13 = 8LL * *(unsigned int *)(a2 + 8);
    if ( v13 )
    {
      memmove(v10, *(const void **)a2, v13);
      v11 = (__int64 *)(a1 + 864);
    }
    v18 = v11;
    v14 = sub_16BD760(v25, v11);
    v15 = *(unsigned int *)(a2 + 8);
    v19 = v14;
    v20 = v16;
    v17 = sub_145CDC0(0x30u, v18);
    if ( v17 )
    {
      *(_QWORD *)(v17 + 32) = v12;
      *(_QWORD *)v17 = 0;
      *(_QWORD *)(v17 + 8) = v19;
      *(_QWORD *)(v17 + 16) = v20;
      *(_DWORD *)(v17 + 24) = 5;
      *(_QWORD *)(v17 + 40) = v15;
    }
    v21 = v17;
    sub_16BDA20(a1 + 816, v17, v24);
    sub_146DBF0(a1, v21);
    result = v21;
  }
  v9 = (_BYTE *)v25[0];
  *(_WORD *)(result + 26) |= a3;
  if ( v9 != v26 )
  {
    v23 = result;
    _libc_free((unsigned __int64)v9);
    return v23;
  }
  return result;
}
