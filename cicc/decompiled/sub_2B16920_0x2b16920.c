// Function: sub_2B16920
// Address: 0x2b16920
//
__int64 __fastcall sub_2B16920(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  bool v4; // r14
  unsigned __int8 **v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned __int64 v11; // r10
  unsigned __int64 v12; // rbx
  _BYTE *v13; // rcx
  int v14; // edx
  _BYTE *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int *v18; // rcx
  __int64 result; // rax
  __int64 v20; // [rsp+0h] [rbp-90h]
  __int64 v21; // [rsp+8h] [rbp-88h]
  __int64 v22; // [rsp+10h] [rbp-80h]
  __int64 v23; // [rsp+18h] [rbp-78h]
  int v24; // [rsp+18h] [rbp-78h]
  _BYTE *v25; // [rsp+20h] [rbp-70h] BYREF
  __int64 v26; // [rsp+28h] [rbp-68h]
  _BYTE v27[96]; // [rsp+30h] [rbp-60h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 8LL * a2);
  if ( *(_BYTE *)v2 == 13 )
    return 0;
  v3 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 8LL * a2);
  v4 = *(_BYTE *)v2 != 41;
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v5 = *(unsigned __int8 ***)(v2 - 8);
  else
    v5 = (unsigned __int8 **)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
  v6 = sub_DFB770(*v5);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(v2 - 8);
  else
    v7 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  v8 = sub_DFB770(*(unsigned __int8 **)(v7 + 32LL * v4));
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
  {
    v9 = *(_QWORD *)(v2 - 8);
    v3 = v9 + 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  }
  else
  {
    v9 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  }
  v26 = 0x600000000LL;
  v10 = v3 - v9;
  v25 = v27;
  v11 = (v3 - v9) >> 5;
  v12 = v11;
  if ( (unsigned __int64)v10 > 0xC0 )
  {
    v22 = v9;
    v20 = v8;
    v21 = v10;
    v24 = v11;
    sub_C8D5F0((__int64)&v25, v27, v11, 8u, v9, v8);
    v15 = v25;
    v14 = v26;
    LODWORD(v11) = v24;
    v9 = v22;
    v10 = v21;
    v8 = v20;
    v13 = &v25[8 * (unsigned int)v26];
  }
  else
  {
    v13 = v27;
    v14 = 0;
    v15 = v27;
  }
  if ( v10 > 0 )
  {
    v16 = 0;
    do
    {
      *(_QWORD *)&v13[v16] = *(_QWORD *)(v9 + 4 * v16);
      v16 += 8;
      --v12;
    }
    while ( v12 );
    v15 = v25;
    v14 = v26;
  }
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(unsigned int **)(a1 + 32);
  LODWORD(v26) = v14 + v11;
  result = sub_DFD800(
             *(_QWORD *)(v17 + 3296),
             **(_DWORD **)(a1 + 16),
             **(_QWORD **)(a1 + 24),
             *v18,
             v6,
             v8,
             v15,
             (unsigned int)(v14 + v11),
             v2,
             0);
  if ( v25 != v27 )
  {
    v23 = result;
    _libc_free((unsigned __int64)v25);
    return v23;
  }
  return result;
}
