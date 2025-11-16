// Function: sub_94D300
// Address: 0x94d300
//
__int64 __fastcall sub_94D300(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int64 *v6; // r14
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // r13
  __m128i *v11; // rax
  unsigned int **v12; // rdi
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v16; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v18[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v6 = *(unsigned __int64 **)(a4 + 16);
  v7 = v6[2];
  v8 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v6, 0, a4);
  v9 = *(__int64 **)(a2 + 32);
  v16 = v8;
  v10 = sub_90A810(v9, a3, (__int64)&v16, 1u);
  v17[0] = sub_92F410(a2, (__int64)v6);
  v11 = sub_92F410(a2, v7);
  v12 = (unsigned int **)(a2 + 48);
  v13 = 0;
  v17[1] = v11;
  v19 = 257;
  if ( v10 )
    v13 = *(_QWORD *)(v10 + 24);
  v14 = sub_921880(v12, v13, v10, (int)v17, 2, (__int64)v18, 0);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v14;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
