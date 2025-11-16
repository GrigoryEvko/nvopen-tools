// Function: sub_21463F0
// Address: 0x21463f0
//
__int64 *__fastcall sub_21463F0(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        int a8)
{
  __int64 v10; // rsi
  char *v11; // rdx
  char v12; // al
  const void **v13; // rdx
  __int64 v14; // r9
  __int64 v15; // r14
  unsigned int *v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rdi
  unsigned __int8 *v20; // rax
  __int64 v21; // r8
  unsigned int v22; // ecx
  _QWORD *v23; // rax
  int v24; // edx
  unsigned int v25; // r9d
  _QWORD *v26; // r15
  int v27; // r8d
  __int64 v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 *v31; // r14
  __int128 v33; // [rsp-10h] [rbp-1B0h]
  int v34; // [rsp+8h] [rbp-198h]
  unsigned int v35; // [rsp+1Ch] [rbp-184h]
  unsigned int v36; // [rsp+1Ch] [rbp-184h]
  __int64 v37; // [rsp+30h] [rbp-170h] BYREF
  int v38; // [rsp+38h] [rbp-168h]
  __int64 v39; // [rsp+40h] [rbp-160h] BYREF
  const void **v40; // [rsp+48h] [rbp-158h]
  __int64 v41; // [rsp+50h] [rbp-150h] BYREF
  int v42; // [rsp+58h] [rbp-148h]
  unsigned int *v43; // [rsp+60h] [rbp-140h] BYREF
  __int64 v44; // [rsp+68h] [rbp-138h]
  _BYTE v45[304]; // [rsp+70h] [rbp-130h] BYREF

  v10 = *(_QWORD *)(a2 + 72);
  v37 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v37, v10, 2);
  v11 = *(char **)(a2 + 40);
  v38 = *(_DWORD *)(a2 + 64);
  v12 = *v11;
  v13 = (const void **)*((_QWORD *)v11 + 1);
  LOBYTE(v39) = v12;
  v40 = v13;
  if ( v12 )
    v14 = word_4310E40[(unsigned __int8)(v12 - 14)];
  else
    v14 = (unsigned int)sub_1F58D30((__int64)&v39);
  v15 = (unsigned int)v14;
  v44 = 0x1000000000LL;
  v16 = (unsigned int *)v45;
  v43 = (unsigned int *)v45;
  if ( (unsigned int)v14 > 0x10 )
  {
    v36 = v14;
    sub_16CD150((__int64)&v43, v45, (unsigned int)v14, 16, a8, v14);
    v16 = v43;
    v14 = v36;
  }
  LODWORD(v44) = v14;
  v17 = (unsigned __int64)&v16[4 * v15];
  if ( v16 != (unsigned int *)v17 )
  {
    do
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = 0;
        v16[2] = 0;
      }
      v16 += 4;
    }
    while ( (unsigned int *)v17 != v16 );
    v17 = (unsigned __int64)v43;
  }
  v18 = *(_QWORD *)(a2 + 32);
  v35 = v14;
  *(_QWORD *)v17 = *(_QWORD *)v18;
  *(_DWORD *)(v17 + 8) = *(_DWORD *)(v18 + 8);
  v19 = *(_QWORD **)(a1 + 8);
  v20 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v43 + 40LL) + 16LL * v43[2]);
  v21 = *((_QWORD *)v20 + 1);
  v22 = *v20;
  v41 = 0;
  v42 = 0;
  v23 = sub_1D2B300(v19, 0x30u, (__int64)&v41, v22, v21, v14);
  v25 = v35;
  v26 = v23;
  v27 = v24;
  if ( v41 )
  {
    v34 = v24;
    sub_161E7C0((__int64)&v41, v41);
    v27 = v34;
    v25 = v35;
  }
  v28 = 16 * v15;
  v29 = 16;
  if ( v25 > 1 )
  {
    do
    {
      v30 = (unsigned __int64)v43;
      *(_QWORD *)&v43[(unsigned __int64)v29 / 4] = v26;
      *(_DWORD *)(v30 + v29 + 8) = v27;
      v29 += 16;
    }
    while ( v28 != v29 );
  }
  *((_QWORD *)&v33 + 1) = (unsigned int)v44;
  *(_QWORD *)&v33 = v43;
  v31 = sub_1D359D0(*(__int64 **)(a1 + 8), 104, (__int64)&v37, v39, v40, 0, a3, a4, a5, v33);
  if ( v43 != (unsigned int *)v45 )
    _libc_free((unsigned __int64)v43);
  if ( v37 )
    sub_161E7C0((__int64)&v37, v37);
  return v31;
}
