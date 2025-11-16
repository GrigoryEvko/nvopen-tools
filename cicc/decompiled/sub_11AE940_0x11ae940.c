// Function: sub_11AE940
// Address: 0x11ae940
//
__int64 __fastcall sub_11AE940(
        const __m128i *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __m128i *a7)
{
  __int64 v10; // rsi
  __int64 v11; // rbx
  unsigned __int8 *v12; // r12
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned int v15; // edx
  int *v16; // rax
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int8 *v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v10 = *(_QWORD *)(a2 - 8);
  else
    v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v11 = v10 + 32LL * a3;
  v12 = *(unsigned __int8 **)v11;
  if ( **(_BYTE **)v11 <= 0x15u )
    goto LABEL_39;
  v13 = *(_DWORD *)(a5 + 8);
  if ( v13 > 0x40 )
  {
    memset(*(void **)a5, 0, 8 * (((unsigned __int64)v13 + 63) >> 6));
    v14 = *(_DWORD *)(a5 + 24);
    if ( v14 <= 0x40 )
    {
LABEL_6:
      *(_QWORD *)(a5 + 16) = 0;
      v15 = *(_DWORD *)(a4 + 8);
      if ( v15 > 0x40 )
        goto LABEL_7;
      goto LABEL_26;
    }
  }
  else
  {
    v14 = *(_DWORD *)(a5 + 24);
    *(_QWORD *)a5 = 0;
    if ( v14 <= 0x40 )
      goto LABEL_6;
  }
  memset(*(void **)(a5 + 16), 0, 8 * (((unsigned __int64)v14 + 63) >> 6));
  v15 = *(_DWORD *)(a4 + 8);
  if ( v15 > 0x40 )
  {
LABEL_7:
    if ( v15 != (unsigned int)sub_C444A0(a4) )
      goto LABEL_8;
LABEL_27:
    v26 = sub_ACA8A0(*((__int64 ***)v12 + 1));
    v27 = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 )
    {
      v28 = *(_QWORD *)(v11 + 8);
      **(_QWORD **)(v11 + 16) = v28;
      if ( v28 )
        *(_QWORD *)(v28 + 16) = *(_QWORD *)(v11 + 16);
    }
    *(_QWORD *)v11 = v26;
    if ( v26 )
    {
      v29 = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v11 + 8) = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = v11 + 8;
      *(_QWORD *)(v11 + 16) = v26 + 16;
      *(_QWORD *)(v26 + 16) = v11;
    }
    if ( *(_BYTE *)v27 <= 0x1Cu )
      return 1;
    v32[0] = v27;
    v30 = a1[2].m128i_i64[1] + 2096;
    sub_11A2F60(v30, v32);
    v31 = *(_QWORD *)(v27 + 16);
    if ( !v31 || *(_QWORD *)(v31 + 8) )
      return 1;
    v32[0] = *(_QWORD *)(v31 + 24);
    sub_11A2F60(v30, v32);
    return 1;
  }
LABEL_26:
  if ( !*(_QWORD *)a4 )
    goto LABEL_27;
LABEL_8:
  if ( *v12 <= 0x1Cu )
  {
LABEL_39:
    sub_9AC0E0((__int64)v12, (unsigned __int64 *)a5, a6, a7);
    return 0;
  }
  v16 = (int *)sub_C94E20((__int64)qword_4F862D0);
  if ( v16 )
    v17 = *v16;
  else
    v17 = qword_4F862D0[2];
  if ( a6 != v17 )
  {
    v18 = *((_QWORD *)v12 + 2);
    if ( !v18 || *(_QWORD *)(v18 + 8) )
      v19 = sub_11A1430((__int64)a1, v12, a4, (__int64 **)a5, a6, a7);
    else
      v19 = sub_11A7600(a1, v12, a4, (unsigned __int64 *)a5, a6, a7);
    if ( v19 )
    {
      v20 = *(unsigned __int8 **)v11;
      if ( **(_BYTE **)v11 <= 0x1Cu || (sub_F54ED0(*(unsigned __int8 **)v11), (v20 = *(unsigned __int8 **)v11) != 0) )
      {
        v21 = *(_QWORD *)(v11 + 8);
        **(_QWORD **)(v11 + 16) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = *(_QWORD *)(v11 + 16);
      }
      *(_QWORD *)v11 = v19;
      v22 = *(_QWORD *)(v19 + 16);
      *(_QWORD *)(v11 + 8) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = v11 + 8;
      *(_QWORD *)(v11 + 16) = v19 + 16;
      *(_QWORD *)(v19 + 16) = v11;
      if ( *v20 > 0x1Cu )
      {
        v32[0] = (__int64)v20;
        v23 = a1[2].m128i_i64[1] + 2096;
        sub_11A2F60(v23, v32);
        v24 = *((_QWORD *)v20 + 2);
        if ( v24 )
        {
          if ( !*(_QWORD *)(v24 + 8) )
          {
            v32[0] = *(_QWORD *)(v24 + 24);
            sub_11A2F60(v23, v32);
          }
        }
      }
      return 1;
    }
  }
  return 0;
}
