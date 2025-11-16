// Function: sub_1AF47C0
// Address: 0x1af47c0
//
__int64 __fastcall sub_1AF47C0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  unsigned __int64 v12; // rbx
  double v13; // xmm4_8
  double v14; // xmm5_8
  unsigned __int64 *v15; // rax
  __int64 v16; // r9
  __int64 v17; // r8
  unsigned int v18; // r12d
  unsigned __int64 v19; // r9
  int v20; // edx
  _QWORD *v21; // rcx
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rdi
  bool v25; // zf
  int v26; // eax
  __int64 v27; // rax
  int v28; // ecx
  _QWORD *v29; // r8
  unsigned int v30; // eax
  __int64 *v31; // rdx
  __int64 v32; // r9
  __int64 v33; // rdi
  int v35; // r10d
  int v36; // edx
  int v37; // r10d
  unsigned __int64 v38; // [rsp+10h] [rbp-160h]
  __int64 v39; // [rsp+18h] [rbp-158h]
  __int64 v40; // [rsp+20h] [rbp-150h] BYREF
  __int64 v41; // [rsp+28h] [rbp-148h]
  _QWORD *v42; // [rsp+30h] [rbp-140h] BYREF
  int v43; // [rsp+38h] [rbp-138h]
  _BYTE *v44; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+B8h] [rbp-B8h]
  _BYTE v46[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v11 = sub_157EB90(a1);
  v40 = 0;
  v41 = 1;
  v12 = sub_1632FA0(v11);
  v15 = (unsigned __int64 *)&v42;
  do
    *v15++ = -8;
  while ( v15 != (unsigned __int64 *)&v44 );
  v16 = *(_QWORD *)(a1 + 40);
  v17 = *(_QWORD *)(a1 + 48);
  v18 = 0;
  v44 = v46;
  v19 = v16 & 0xFFFFFFFFFFFFFFF8LL;
  v45 = 0x1000000000LL;
  if ( v17 == v19 )
    goto LABEL_24;
  do
  {
    while ( 1 )
    {
      v24 = v17 - 24;
      v25 = v17 == 0;
      v17 = *(_QWORD *)(v17 + 8);
      if ( v25 )
        v24 = 0;
      if ( (v41 & 1) != 0 )
      {
        v20 = 15;
        v21 = &v42;
      }
      else
      {
        v21 = v42;
        v20 = v43 - 1;
        if ( !v43 )
          goto LABEL_12;
      }
      v22 = v20 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v23 = v21[v22];
      if ( v24 != v23 )
        break;
LABEL_7:
      if ( v17 == v19 )
        goto LABEL_13;
    }
    v35 = 1;
    while ( v23 != -8 )
    {
      v22 = v20 & (v35 + v22);
      v23 = v21[v22];
      if ( v24 == v23 )
        goto LABEL_7;
      ++v35;
    }
LABEL_12:
    v38 = v19;
    v39 = v17;
    v26 = sub_1AF45F0(v24, (__int64)&v40, v12, a2, a3, a4, a5, a6, v13, v14, a9, a10);
    v17 = v39;
    v19 = v38;
    v18 |= v26;
  }
  while ( v39 != v38 );
LABEL_13:
  v27 = (unsigned int)v45;
  if ( (_DWORD)v45 )
  {
    while ( 1 )
    {
      v33 = *(_QWORD *)&v44[8 * v27 - 8];
      if ( (v41 & 1) != 0 )
        break;
      v29 = v42;
      v28 = v43 - 1;
      if ( v43 )
        goto LABEL_16;
LABEL_18:
      LODWORD(v45) = v45 - 1;
      v18 |= sub_1AF45F0(v33, (__int64)&v40, v12, a2, a3, a4, a5, a6, v13, v14, a9, a10);
      v27 = (unsigned int)v45;
      if ( !(_DWORD)v45 )
        goto LABEL_22;
    }
    v28 = 15;
    v29 = &v42;
LABEL_16:
    v30 = v28 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
    v31 = &v29[v30];
    v32 = *v31;
    if ( *v31 == v33 )
    {
LABEL_17:
      *v31 = -16;
      ++HIDWORD(v41);
      LODWORD(v41) = (2 * ((unsigned int)v41 >> 1) - 2) | v41 & 1;
    }
    else
    {
      v36 = 1;
      while ( v32 != -8 )
      {
        v37 = v36 + 1;
        v30 = v28 & (v36 + v30);
        v31 = &v29[v30];
        v32 = *v31;
        if ( v33 == *v31 )
          goto LABEL_17;
        v36 = v37;
      }
    }
    goto LABEL_18;
  }
LABEL_22:
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
LABEL_24:
  if ( (v41 & 1) == 0 )
    j___libc_free_0(v42);
  return v18;
}
