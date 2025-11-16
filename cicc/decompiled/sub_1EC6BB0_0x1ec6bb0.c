// Function: sub_1EC6BB0
// Address: 0x1ec6bb0
//
__int64 __fastcall sub_1EC6BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  double *v8; // r15
  size_t v9; // rax
  const void *v10; // r8
  __int64 v11; // r9
  int v12; // r11d
  __int64 v13; // r10
  __int64 v14; // rax
  unsigned __int16 *v15; // rdi
  unsigned int v16; // r15d
  unsigned __int16 *v17; // rsi
  __int64 v19; // rdx
  unsigned __int16 *v20; // rax
  __int64 v21; // rcx
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rdi
  _QWORD *v25; // rax
  unsigned __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // r15
  unsigned int src; // [rsp+0h] [rbp-70h]
  unsigned int v31; // [rsp+18h] [rbp-58h]
  int v32; // [rsp+18h] [rbp-58h]
  unsigned int v34; // [rsp+2Ch] [rbp-44h] BYREF
  int v35[2]; // [rsp+30h] [rbp-40h] BYREF
  __int64 v36; // [rsp+38h] [rbp-38h] BYREF

  v31 = unk_4F9E388;
  v8 = (double *)strlen("Register Allocation");
  v9 = strlen("regalloc");
  sub_16D8B50(
    (__m128i **)v35,
    (unsigned __int8 *)"evict",
    5u,
    (__int64)"Evict",
    5,
    v31,
    "regalloc",
    v9,
    "Register Allocation",
    v8);
  v36 = 0xFFFFFFFFLL;
  v32 = *(_DWORD *)(a3 + 56);
  if ( a5 != -1 )
  {
    v22 = *(_DWORD *)(a2 + 112);
    v23 = *(_QWORD *)(a1 + 248);
    LODWORD(v36) = 0;
    v24 = a1 + 280;
    v25 = (_QWORD *)(*(_QWORD *)(v23 + 24) + 16LL * (v22 & 0x7FFFFFFF));
    HIDWORD(v36) = *(_DWORD *)(a2 + 116);
    v26 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
    v27 = *(_QWORD *)(a1 + 280) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v26 + 24LL);
    if ( *(_DWORD *)(a1 + 288) != *(_DWORD *)v27 )
    {
      sub_1ED7890(v24);
      v24 = a1 + 280;
    }
    if ( a5 <= *(unsigned __int8 *)(v27 + 9) )
    {
LABEL_11:
      src = 0;
      goto LABEL_12;
    }
    if ( a5 <= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 696) + 232LL)
                         + 8LL * *(unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 2LL * *(_QWORD *)(a3 + 56) - 2)) )
    {
      v28 = *(_QWORD *)(a1 + 280) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v26 + 24LL);
      if ( *(_DWORD *)(a1 + 288) != *(_DWORD *)v28 )
        sub_1ED7890(v24);
      v32 = *(unsigned __int16 *)(v28 + 10);
    }
  }
  src = 0;
  v11 = (unsigned int)-*(_DWORD *)(a3 + 8);
  *(_DWORD *)(a3 + 64) = v11;
  while ( (int)v11 < 0 )
  {
    v19 = *(unsigned int *)(a3 + 8);
    v20 = *(unsigned __int16 **)a3;
    *(_DWORD *)(a3 + 64) = v11 + 1;
    v11 = v19 + (int)v11;
    v16 = v20[v11];
LABEL_16:
    if ( !v16 )
      goto LABEL_10;
    v21 = a5;
    if ( a5 <= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 696) + 232LL) + 8LL * v16)
      || a5 == 1
      && v16 < *(_DWORD *)(a1 + 328)
      && *(_WORD *)(*(_QWORD *)(a1 + 320) + 2LL * v16)
      && !(unsigned __int8)sub_2103340(*(_QWORD *)(a1 + 272))
      || (int)sub_21038C0(*(_QWORD *)(a1 + 272), a2, v16, v21) > 1
      || !(unsigned __int8)sub_1EBD330(a1, a2, v16, 0, (float *)&v36) )
    {
      v11 = *(unsigned int *)(a3 + 64);
    }
    else
    {
      v11 = *(unsigned int *)(a3 + 64);
      src = v16;
      if ( (int)v11 <= 0 )
        goto LABEL_23;
    }
  }
  if ( !*(_BYTE *)(a3 + 68) )
  {
    v12 = v32;
    if ( !v32 )
      v12 = *(_DWORD *)(a3 + 56);
    v13 = 2LL * (int)v11;
    while ( v12 > (int)v11 )
    {
      v14 = *(_QWORD *)(a3 + 48);
      v15 = *(unsigned __int16 **)a3;
      *(_DWORD *)(a3 + 64) = v11 + 1;
      v16 = *(unsigned __int16 *)(v14 + v13);
      v17 = &v15[*(unsigned int *)(a3 + 8)];
      v34 = v16;
      if ( v17 == sub_1EBB4B0(v15, (__int64)v17, (int *)&v34) )
        goto LABEL_16;
    }
  }
LABEL_10:
  if ( !src )
    goto LABEL_11;
LABEL_23:
  sub_1EC63D0(a1, a2, src, a4, v10, v11);
LABEL_12:
  if ( *(_QWORD *)v35 )
    sub_16D7950(*(__int64 *)v35);
  return src;
}
