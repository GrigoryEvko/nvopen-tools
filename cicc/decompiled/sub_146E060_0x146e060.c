// Function: sub_146E060
// Address: 0x146e060
//
__int64 __fastcall sub_146E060(__int64 a1, __int64 a2, __int16 a3)
{
  __int64 v4; // rbx
  __int64 *v5; // r13
  __int64 v6; // rsi
  __int64 result; // rax
  _BYTE *v8; // rdi
  void *v9; // rax
  __int64 *v10; // r10
  void *v11; // r13
  size_t v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // [rsp+8h] [rbp-F8h]
  __int64 v18; // [rsp+10h] [rbp-F0h]
  __int64 v19; // [rsp+18h] [rbp-E8h]
  __int64 v20; // [rsp+18h] [rbp-E8h]
  __int64 v22; // [rsp+28h] [rbp-D8h]
  __int64 v23; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD v24[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v25[176]; // [rsp+50h] [rbp-B0h] BYREF

  v24[1] = 0x2000000000LL;
  v24[0] = v25;
  sub_16BD3E0(v24, 4);
  v4 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v4 )
  {
    v5 = *(__int64 **)a2;
    do
    {
      v6 = *v5++;
      sub_16BD4C0(v24, v6);
    }
    while ( (__int64 *)v4 != v5 );
  }
  v23 = 0;
  result = sub_16BDDE0(a1 + 816, v24, &v23);
  if ( !result )
  {
    v9 = (void *)sub_145CBF0((__int64 *)(a1 + 864), 8LL * *(unsigned int *)(a2 + 8), 8);
    v10 = (__int64 *)(a1 + 864);
    v11 = v9;
    v12 = 8LL * *(unsigned int *)(a2 + 8);
    if ( v12 )
    {
      memmove(v9, *(const void **)a2, v12);
      v10 = (__int64 *)(a1 + 864);
    }
    v17 = v10;
    v13 = sub_16BD760(v24, v10);
    v14 = *(unsigned int *)(a2 + 8);
    v18 = v13;
    v19 = v15;
    v16 = sub_145CDC0(0x30u, v17);
    if ( v16 )
    {
      *(_QWORD *)(v16 + 32) = v11;
      *(_QWORD *)v16 = 0;
      *(_QWORD *)(v16 + 8) = v18;
      *(_QWORD *)(v16 + 16) = v19;
      *(_DWORD *)(v16 + 24) = 4;
      *(_QWORD *)(v16 + 40) = v14;
    }
    v20 = v16;
    sub_16BDA20(a1 + 816, v16, v23);
    sub_146DBF0(a1, v20);
    result = v20;
  }
  v8 = (_BYTE *)v24[0];
  *(_WORD *)(result + 26) |= a3;
  if ( v8 != v25 )
  {
    v22 = result;
    _libc_free((unsigned __int64)v8);
    return v22;
  }
  return result;
}
