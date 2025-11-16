// Function: sub_33E4BC0
// Address: 0x33e4bc0
//
_QWORD *__fastcall sub_33E4BC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const void *a4,
        __int64 a5,
        char a6,
        const void *a7,
        __int64 a8,
        __int64 *a9,
        int a10,
        char a11)
{
  __int64 *v12; // r13
  __int64 *v13; // r9
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  __int64 v17; // rdx
  size_t v18; // r8
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rbx
  unsigned __int64 v23; // rax
  __int64 v24; // rsi
  size_t v26; // [rsp+0h] [rbp-70h]
  size_t v31; // [rsp+28h] [rbp-48h]
  __int64 v32[7]; // [rsp+38h] [rbp-38h] BYREF

  v12 = *(__int64 **)(a1 + 720);
  v13 = v12;
  v14 = *a9;
  v32[0] = v14;
  if ( v14 )
  {
    sub_B96E90((__int64)v32, v14, 1);
    v13 = *(__int64 **)(a1 + 720);
  }
  v15 = (_QWORD *)sub_A777F0(0x40u, v13);
  v16 = v15;
  if ( v15 )
  {
    *v15 = a5;
    v17 = *v12;
    v18 = 24 * a5;
    v12[10] += 24 * a5;
    v19 = (v17 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v20 = 24 * a5 + v19;
    if ( v12[1] >= v20 && v17 )
    {
      *v12 = v20;
    }
    else
    {
      v19 = sub_9D1E70((__int64)v12, 24 * a5, 24 * a5, 3);
      v18 = 24 * a5;
    }
    v16[1] = v19;
    v16[2] = a8;
    v21 = *v12;
    v22 = 8 * a8;
    v12[10] += 8 * a8;
    v23 = (v21 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v12[1] >= 8 * a8 + v23 && v21 )
    {
      *v12 = v22 + v23;
    }
    else
    {
      v26 = v18;
      v23 = sub_9D1E70((__int64)v12, v22, v22, 3);
      v18 = v26;
    }
    v16[3] = v23;
    v24 = v32[0];
    v16[4] = a2;
    v16[6] = v24;
    v16[5] = a3;
    if ( v24 )
    {
      v31 = v18;
      sub_B96E90((__int64)(v16 + 6), v24, 1);
      v18 = v31;
    }
    *((_BYTE *)v16 + 61) = a11;
    *((_DWORD *)v16 + 14) = a10;
    *((_BYTE *)v16 + 60) = a6;
    *((_WORD *)v16 + 31) = 0;
    if ( v18 )
      memmove((void *)v16[1], a4, v18);
    if ( v22 )
      memmove((void *)v16[3], a7, v22);
  }
  if ( v32[0] )
    sub_B91220((__int64)v32, v32[0]);
  return v16;
}
