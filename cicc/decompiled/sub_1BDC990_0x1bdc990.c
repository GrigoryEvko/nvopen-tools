// Function: sub_1BDC990
// Address: 0x1bdc990
//
__int64 __fastcall sub_1BDC990(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13)
{
  int v14; // r13d
  __int64 v17; // r9
  int v18; // eax
  __int64 v19; // r10
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  char v22; // al
  unsigned int v23; // r12d
  __int64 v25; // rax
  __int64 *v26; // rsi
  __int64 v27; // rax
  __int64 *v28; // r9
  __int64 *v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 *v34; // rcx
  __int64 *v35; // rax
  __int64 v36; // [rsp+8h] [rbp-E8h]
  __int64 v37; // [rsp+18h] [rbp-D8h]
  __int64 v38; // [rsp+18h] [rbp-D8h]
  _BYTE v39[8]; // [rsp+28h] [rbp-C8h] BYREF
  __int64 *v40; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+38h] [rbp-B8h]
  _BYTE v42[176]; // [rsp+40h] [rbp-B0h] BYREF

  v14 = 0;
  v41 = 0x1000000000LL;
  v17 = *(_QWORD *)(a1 + 8);
  v40 = (__int64 *)v42;
  while ( 1 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) == 13 )
    {
      v37 = v17;
      v18 = sub_14A3470(v17);
      v17 = v37;
      v14 += v18;
    }
    v19 = *(_QWORD *)(a2 - 48);
    v20 = (unsigned int)v41;
    if ( (unsigned int)v41 >= HIDWORD(v41) )
    {
      v36 = *(_QWORD *)(a2 - 48);
      v38 = v17;
      sub_16CD150((__int64)&v40, v42, 0, 8, a13, v17);
      v20 = (unsigned int)v41;
      v19 = v36;
      v17 = v38;
    }
    v40[v20] = v19;
    v21 = (unsigned int)(v41 + 1);
    LODWORD(v41) = v41 + 1;
    a2 = *(_QWORD *)(a2 - 72);
    v22 = *(_BYTE *)(a2 + 16);
    if ( v22 == 9 )
      break;
    if ( v22 == 84 )
    {
      v25 = *(_QWORD *)(a2 + 8);
      if ( v25 )
      {
        if ( !*(_QWORD *)(v25 + 8) )
          continue;
      }
    }
    goto LABEL_8;
  }
  v26 = v40;
  v27 = v21;
  v28 = &v40[v21];
  if ( v40 != v28 )
  {
    v29 = v28 - 1;
    if ( v40 < v28 - 1 )
    {
      do
      {
        v30 = *v26;
        v31 = *v29;
        ++v26;
        --v29;
        *(v26 - 1) = v31;
        v29[1] = v30;
      }
      while ( v26 < v29 );
      v21 = (unsigned int)v41;
      v26 = v40;
      v27 = (unsigned int)v41;
      v28 = &v40[v27];
    }
  }
  v32 = (v27 * 8) >> 3;
  v33 = (v27 * 8) >> 5;
  if ( v33 )
  {
    v34 = v26;
    v35 = &v26[4 * v33];
    while ( *(_BYTE *)(*v34 + 16) == 83 )
    {
      if ( *(_BYTE *)(v34[1] + 16) != 83 )
      {
        ++v34;
        break;
      }
      if ( *(_BYTE *)(v34[2] + 16) != 83 )
      {
        v34 += 2;
        break;
      }
      if ( *(_BYTE *)(v34[3] + 16) != 83 )
      {
        v34 += 3;
        break;
      }
      v34 += 4;
      if ( v35 == v34 )
      {
        v32 = v28 - v34;
        goto LABEL_25;
      }
    }
LABEL_22:
    if ( v34 != v28 )
      goto LABEL_23;
    goto LABEL_29;
  }
  v34 = v26;
LABEL_25:
  if ( v32 != 2 )
  {
    if ( v32 != 3 )
    {
      if ( v32 != 1 )
        goto LABEL_29;
      goto LABEL_28;
    }
    if ( *(_BYTE *)(*v34 + 16) != 83 )
      goto LABEL_22;
    ++v34;
  }
  if ( *(_BYTE *)(*v34 + 16) != 83 )
    goto LABEL_22;
  ++v34;
LABEL_28:
  if ( *(_BYTE *)(*v34 + 16) != 83 )
    goto LABEL_22;
LABEL_29:
  sub_1BBA340((__int64)v39, v26, v21);
  if ( v39[4] )
  {
LABEL_8:
    v23 = 0;
    goto LABEL_9;
  }
  v26 = v40;
  v21 = (unsigned int)v41;
LABEL_23:
  v23 = sub_1BDB410(a1, (__int64 ***)v26, v21, a12, v14, 0, a3, a4, a5, a6, a7, a8, a9, a10);
LABEL_9:
  if ( v40 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v40);
  return v23;
}
