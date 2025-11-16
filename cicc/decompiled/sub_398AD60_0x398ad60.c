// Function: sub_398AD60
// Address: 0x398ad60
//
__int64 __fastcall sub_398AD60(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // r9
  __int64 v7; // r10
  unsigned int v8; // r15d
  unsigned int v9; // r8d
  unsigned int v10; // edx
  _QWORD *v11; // rax
  char v12; // al
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  const __m128i *v15; // roff
  __int64 v16; // rax
  unsigned int v18; // [rsp+Ch] [rbp-F4h]
  __int64 v19; // [rsp+10h] [rbp-F0h]
  __int64 v20; // [rsp+18h] [rbp-E8h]
  _QWORD v21[3]; // [rsp+20h] [rbp-E0h] BYREF
  char v22; // [rsp+38h] [rbp-C8h]
  _OWORD v23[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v24[8]; // [rsp+60h] [rbp-A0h] BYREF
  char *v25; // [rsp+68h] [rbp-98h]
  char v26; // [rsp+78h] [rbp-88h] BYREF

  v5 = *a3;
  v6 = a1[1];
  v7 = *a1;
  v8 = *(_DWORD *)(*a3 + 1464);
  v9 = *(_DWORD *)(*a3 + 1192);
  v10 = *(_DWORD *)(*a3 + 152);
  if ( v10 >= *(_DWORD *)(v5 + 156) )
  {
    v18 = v9;
    v19 = *a1;
    v20 = a1[1];
    sub_16CD150(v5 + 144, (const void *)(v5 + 160), 0, 32, v9, v6);
    v10 = *(_DWORD *)(v5 + 152);
    v9 = v18;
    v7 = v19;
    v6 = v20;
  }
  v11 = (_QWORD *)(*(_QWORD *)(v5 + 144) + 32LL * v10);
  if ( v11 )
  {
    *v11 = v7;
    v11[2] = v9;
    v11[1] = v6;
    v11[3] = v8;
    v10 = *(_DWORD *)(v5 + 152);
  }
  v12 = *(_BYTE *)(v5 + 2496);
  *(_DWORD *)(v5 + 152) = v10 + 1;
  v22 = v12;
  v21[0] = &unk_4A3F938;
  v21[1] = v5 + 1184;
  v21[2] = v5 + 1456;
  v13 = sub_3971A70(a2);
  sub_39886F0((__int64)v24, v13, (__int64)v21, a2);
  v14 = *(unsigned int *)(a4 + 8);
  v15 = (const __m128i *)a1[2];
  v23[0] = _mm_loadu_si128(v15);
  v16 = *(_QWORD *)(a4 + 8 * (4 - v14));
  v23[1] = _mm_loadu_si128(v15 + 1);
  *(_QWORD *)&v23[0] = v16;
  sub_3984CB0(a2, 0, (__int64 *)v23, (__int64)v24);
  sub_399FD30(v24);
  if ( v25 != &v26 )
    _libc_free((unsigned __int64)v25);
  return sub_39C2CC0(v5);
}
