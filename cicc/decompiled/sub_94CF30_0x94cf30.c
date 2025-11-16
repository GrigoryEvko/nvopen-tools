// Function: sub_94CF30
// Address: 0x94cf30
//
__int64 __fastcall sub_94CF30(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // r13
  __m128i *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned int **v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  _BYTE *v16; // rdi
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-118h]
  __m128i *v20; // [rsp+8h] [rbp-118h]
  _BYTE v22[32]; // [rsp+20h] [rbp-100h] BYREF
  __int16 v23; // [rsp+40h] [rbp-E0h]
  _BYTE *v24; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+58h] [rbp-C8h]
  _BYTE v26[64]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE *i; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v28; // [rsp+A8h] [rbp-78h]
  _BYTE v29[112]; // [rsp+B0h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a3 + 16);
  v25 = 0x800000000LL;
  v28 = 0x800000000LL;
  v24 = v26;
  for ( i = v29; v6; LODWORD(v28) = v28 + 1 )
  {
    v7 = sub_92F410(a2, v6);
    v8 = (unsigned int)v25;
    if ( (unsigned __int64)(unsigned int)v25 + 1 > HIDWORD(v25) )
    {
      v20 = v7;
      sub_C8D5F0(&v24, v26, (unsigned int)v25 + 1LL, 8);
      v8 = (unsigned int)v25;
      v7 = v20;
    }
    *(_QWORD *)&v24[8 * v8] = v7;
    LODWORD(v25) = v25 + 1;
    v9 = v7->m128i_i64[1];
    v10 = (unsigned int)v28;
    v11 = (unsigned int)v28 + 1LL;
    if ( v11 > HIDWORD(v28) )
    {
      v19 = v9;
      sub_C8D5F0(&i, v29, v11, 8);
      v10 = (unsigned int)v28;
      v9 = v19;
    }
    *(_QWORD *)&i[8 * v10] = v9;
    v6 = *(_QWORD *)(v6 + 16);
  }
  v12 = (unsigned int **)(a2 + 48);
  v13 = sub_90A810(*(__int64 **)(a2 + 32), a4, 0, 0);
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v13 + 24) + 16LL) + 8LL) == 7 )
  {
    v23 = 257;
    v14 = *(_QWORD *)(v13 + 24);
    sub_921880(v12, v14, v13, (int)v24, v25, (__int64)v22, 0);
    v18 = sub_BCB2D0(*(_QWORD *)(a2 + 40));
    v15 = sub_AD6530(v18);
  }
  else
  {
    v23 = 257;
    v14 = *(_QWORD *)(v13 + 24);
    v15 = sub_921880(v12, v14, v13, (int)v24, v25, (__int64)v22, 0);
  }
  v16 = i;
  *(_QWORD *)a1 = v15;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v16 != v29 )
    _libc_free(v16, v14);
  if ( v24 != v26 )
    _libc_free(v24, v14);
  return a1;
}
