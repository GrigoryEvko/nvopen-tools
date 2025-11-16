// Function: sub_DE5430
// Address: 0xde5430
//
__int64 __fastcall sub_DE5430(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rdx
  __int64 v9; // rdi
  bool v10; // zf
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 result; // rax
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 *v21; // rcx
  unsigned int v22; // esi
  __int64 **v23; // rax
  __int64 *v24; // r8
  __int64 *v25; // rax
  int v26; // eax
  int v27; // r10d
  __int64 v28; // [rsp+8h] [rbp-88h] BYREF
  char v29[16]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v30; // [rsp+20h] [rbp-70h]
  char v31; // [rsp+30h] [rbp-60h]
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+48h] [rbp-48h]
  __int64 v34; // [rsp+50h] [rbp-40h]
  __int64 v35; // [rsp+58h] [rbp-38h]
  __int64 v36; // [rsp+60h] [rbp-30h]
  __int64 v37; // [rsp+68h] [rbp-28h]

  v6 = a2;
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_BYTE *)(v9 + 28) == 0;
  v11 = *(_QWORD *)(*(_QWORD *)(v8 - 8) + 32LL * *(unsigned int *)(v8 + 72) + 8LL * a2);
  v28 = v11;
  if ( v10 )
    goto LABEL_9;
  v12 = *(_QWORD **)(v9 + 8);
  a4 = *(unsigned int *)(v9 + 20);
  v8 = (__int64)&v12[a4];
  if ( v12 != (_QWORD *)v8 )
  {
    while ( v11 != *v12 )
    {
      if ( (_QWORD *)v8 == ++v12 )
        goto LABEL_8;
    }
    return 0;
  }
LABEL_8:
  if ( (unsigned int)a4 >= *(_DWORD *)(v9 + 16) )
  {
LABEL_9:
    sub_C8CC70(v9, v11, v8, a4, a5, a6);
    if ( !v14 )
      return 0;
  }
  else
  {
    *(_DWORD *)(v9 + 20) = a4 + 1;
    *(_QWORD *)v8 = v11;
    ++*(_QWORD *)v9;
  }
  v15 = *(_QWORD *)(a1 + 24);
  v16 = *(_QWORD *)(a1 + 16);
  v32 = 0;
  v37 = v15;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  sub_DB7320((__int64)v29, v16, &v28, (__int64)&v32);
  v17 = v30;
  sub_C7D6A0(v33, 16LL * (unsigned int)v35, 8);
  if ( v31 )
    sub_DE2750(
      *(__int64 **)(a1 + 24),
      v17 + 8,
      *(_QWORD *)(*(_QWORD *)a1 + 40LL),
      v28,
      *(_QWORD *)(a1 + 8),
      (__int64 *)(unsigned int)(**(_DWORD **)(a1 + 32) + 1));
  if ( !*(_DWORD *)(v17 + 24) )
    return 0;
  v18 = sub_DD8400(*(_QWORD *)(a1 + 24), *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 - 8LL) + 32 * v6));
  v19 = *(unsigned int *)(v17 + 32);
  v20 = *(_QWORD *)(v17 + 16);
  v21 = v18;
  if ( !(_DWORD)v19 )
    return 0;
  v22 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  v23 = (__int64 **)(v20 + 16LL * v22);
  v24 = *v23;
  if ( v21 != *v23 )
  {
    v26 = 1;
    while ( v24 != (__int64 *)-4096LL )
    {
      v27 = v26 + 1;
      v22 = (v19 - 1) & (v26 + v22);
      v23 = (__int64 **)(v20 + 16LL * v22);
      v24 = *v23;
      if ( v21 == *v23 )
        goto LABEL_15;
      v26 = v27;
    }
    return 0;
  }
LABEL_15:
  if ( v23 == (__int64 **)(v20 + 16 * v19) )
    return 0;
  v25 = v23[1];
  if ( !v25 )
    return 0;
  if ( (unsigned __int16)(*((_WORD *)v25 + 12) - 9) > 3u )
    return 0;
  result = *(_QWORD *)v25[4];
  if ( *(_WORD *)(result + 24) )
    return 0;
  return result;
}
