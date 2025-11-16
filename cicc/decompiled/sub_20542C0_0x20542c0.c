// Function: sub_20542C0
// Address: 0x20542c0
//
__int64 *__fastcall sub_20542C0(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 *v15; // rsi
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 *v19; // rax
  unsigned int v20; // edx
  __int64 v21; // r8
  __int64 *v22; // r14
  __int64 v23; // rcx
  int v25; // edx
  int v26; // r9d
  __int64 *v27; // [rsp+0h] [rbp-110h]
  int v28; // [rsp+8h] [rbp-108h]
  __int64 v29; // [rsp+8h] [rbp-108h]
  __int64 *v30; // [rsp+10h] [rbp-100h] BYREF
  int v31; // [rsp+18h] [rbp-F8h]
  __int64 v32; // [rsp+20h] [rbp-F0h] BYREF
  int v33; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v34[2]; // [rsp+30h] [rbp-E0h] BYREF
  char v35; // [rsp+40h] [rbp-D0h] BYREF
  char *v36; // [rsp+80h] [rbp-90h]
  char v37; // [rsp+90h] [rbp-80h] BYREF
  char *v38; // [rsp+98h] [rbp-78h]
  char v39; // [rsp+A8h] [rbp-68h] BYREF
  char *v40; // [rsp+B8h] [rbp-58h]
  char v41; // [rsp+C8h] [rbp-48h] BYREF

  v7 = *(_QWORD *)(a1 + 712);
  v8 = *(unsigned int *)(v7 + 232);
  if ( !(_DWORD)v8 )
    return 0;
  v10 = *(_QWORD *)(v7 + 216);
  v11 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( a2 != *v12 )
  {
    v25 = 1;
    while ( v13 != -8 )
    {
      v26 = v25 + 1;
      v11 = (v8 - 1) & (v25 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_3;
      v25 = v26;
    }
    return 0;
  }
LABEL_3:
  if ( v12 == (__int64 *)(v10 + 16 * v8) )
    return 0;
  v28 = *((_DWORD *)v12 + 2);
  sub_2043DE0((__int64)&v32, a2);
  v14 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  sub_204E3C0(
    (__int64)v34,
    *(_QWORD *)(*(_QWORD *)(a1 + 552) + 48LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL),
    v14,
    v28,
    a3,
    (unsigned int *)&v32);
  v15 = *(__int64 **)(a1 + 552);
  v16 = *(_DWORD *)(a1 + 536);
  v31 = 0;
  v32 = 0;
  v33 = v16;
  v30 = v15 + 11;
  v17 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    if ( &v32 != (__int64 *)(v17 + 48) )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v32 = v18;
      if ( v18 )
      {
        sub_1623A60((__int64)&v32, v18, 2);
        v15 = *(__int64 **)(a1 + 552);
      }
    }
  }
  v19 = sub_204EDE0((__int64)v34, v15, *(_QWORD *)(a1 + 712), (__int64)&v32, (__int64 *)&v30, 0, a4, a5, a6, a2);
  v21 = (__int64)v19;
  v22 = v19;
  v23 = v20;
  if ( v32 )
  {
    v27 = v19;
    v29 = v20;
    sub_161E7C0((__int64)&v32, v32);
    v21 = (__int64)v27;
    v23 = v29;
  }
  sub_20540C0(a1, a2, v21, v23);
  if ( v40 != &v41 )
    _libc_free((unsigned __int64)v40);
  if ( v38 != &v39 )
    _libc_free((unsigned __int64)v38);
  if ( v36 != &v37 )
    _libc_free((unsigned __int64)v36);
  if ( (char *)v34[0] != &v35 )
    _libc_free(v34[0]);
  return v22;
}
