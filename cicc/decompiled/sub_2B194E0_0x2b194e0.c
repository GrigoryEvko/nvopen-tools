// Function: sub_2B194E0
// Address: 0x2b194e0
//
__int64 __fastcall sub_2B194E0(__int64 a1, __int64 a2)
{
  unsigned int v3; // edx
  int v4; // r13d
  unsigned int v5; // r14d
  unsigned int v6; // r15d
  __int64 *v7; // rdi
  unsigned int v8; // r13d
  unsigned int v9; // eax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 *v12; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 *v16; // rbx
  unsigned __int64 v17; // [rsp+0h] [rbp-E0h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-D8h]
  __m128i v19; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v20; // [rsp+20h] [rbp-C0h]
  __int64 v21; // [rsp+28h] [rbp-B8h]
  __int64 v22; // [rsp+30h] [rbp-B0h]
  __int64 v23; // [rsp+38h] [rbp-A8h]
  __int64 v24; // [rsp+40h] [rbp-A0h]
  __int64 v25; // [rsp+48h] [rbp-98h]
  __int16 v26; // [rsp+50h] [rbp-90h]
  __m128i v27; // [rsp+60h] [rbp-80h] BYREF
  __int64 v28; // [rsp+70h] [rbp-70h]
  __int64 v29; // [rsp+78h] [rbp-68h]
  __int64 v30; // [rsp+80h] [rbp-60h]
  __int64 v31; // [rsp+88h] [rbp-58h]
  __int64 v32; // [rsp+90h] [rbp-50h]
  __int64 v33; // [rsp+98h] [rbp-48h]
  __int16 v34; // [rsp+A0h] [rbp-40h]

  v4 = **(_DWORD **)(a1 + 8);
  v18 = **(_DWORD **)a1;
  v3 = v18;
  v5 = v18 - v4;
  v6 = v4 - 1;
  if ( v18 > 0x40 )
  {
    sub_C43690((__int64)&v17, 0, 0);
    v3 = v18;
  }
  else
  {
    v17 = 0;
  }
  if ( v6 != v3 )
  {
    if ( v6 > 0x3F || v3 > 0x40 )
      sub_C43C90(&v17, v6, v3);
    else
      v17 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v4 + 63 - (unsigned __int8)v3) << v6;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v8 = 0;
  v9 = sub_9AF8B0(
         *v7,
         *(_QWORD *)(*(_QWORD *)(a1 + 16) + 3344LL),
         0,
         *(_QWORD *)(*(_QWORD *)(a1 + 16) + 3328LL),
         0,
         *(_QWORD *)(*(_QWORD *)(a1 + 16) + 3320LL),
         1);
  if ( v5 <= v9 )
  {
    if ( v5 == v9 )
      goto LABEL_19;
    v10 = *(_QWORD *)(a1 + 16);
    v20 = 0;
    v11 = *(_QWORD *)(v10 + 3344);
    v21 = 0;
    v22 = 0;
    v19 = (__m128i)v11;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 257;
    v12 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
        ? *(__int64 **)(a2 - 8)
        : (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v8 = 1;
    if ( (unsigned __int8)sub_9AC470(*v12, &v19, 0) )
    {
LABEL_19:
      v14 = *(_QWORD *)(a1 + 16);
      v28 = 0;
      v15 = *(_QWORD *)(v14 + 3344);
      v29 = 0;
      v30 = 0;
      v27 = (__m128i)v15;
      v31 = 0;
      v32 = 0;
      v33 = 0;
      v34 = 257;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v16 = *(__int64 **)(a2 - 8);
      else
        v16 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v8 = sub_9AC230(*v16, (__int64)&v17, &v27, 0);
    }
  }
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return v8;
}
