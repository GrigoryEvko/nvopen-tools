// Function: sub_CFCC00
// Address: 0xcfcc00
//
__int64 __fastcall sub_CFCC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned __int8 *v8; // rsi
  _BYTE *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r10
  unsigned int v13; // esi
  __int64 v14; // rdx
  __int64 v15; // r12
  _QWORD *v16; // r12
  _QWORD *v17; // rsi
  char v18; // r15
  char v19; // di
  char v20; // al
  __int64 v21; // rax
  __int64 result; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  _QWORD *v25; // r15
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 *v28; // r12
  __int64 v29; // rax
  bool v30; // zf
  __int64 v31; // rdi
  bool v32; // al
  int v33; // edx
  int v34; // r15d
  __int64 v35; // [rsp+18h] [rbp-288h]
  _BYTE *v36; // [rsp+20h] [rbp-280h]
  __int64 v37; // [rsp+28h] [rbp-278h]
  __int64 v38; // [rsp+28h] [rbp-278h]
  __int64 v39; // [rsp+28h] [rbp-278h]
  _QWORD v40[2]; // [rsp+38h] [rbp-268h] BYREF
  __int64 v41; // [rsp+48h] [rbp-258h]
  __int64 v42; // [rsp+50h] [rbp-250h]
  _BYTE *v43; // [rsp+60h] [rbp-240h] BYREF
  __int64 v44; // [rsp+68h] [rbp-238h]
  _BYTE v45[560]; // [rsp+70h] [rbp-230h] BYREF

  v8 = *(unsigned __int8 **)(a1 + 8);
  v43 = v45;
  v44 = 0x1000000000LL;
  sub_CFC6F0(a2, v8, (__int64)&v43, a4, a5, a6);
  v9 = v43;
  v36 = &v43[32 * (unsigned int)v44];
  if ( v43 == v36 )
    goto LABEL_17;
  do
  {
    while ( 1 )
    {
      v10 = *(unsigned int *)(a1 + 184);
      if ( !(_DWORD)v10 )
        goto LABEL_16;
      v11 = *((_QWORD *)v9 + 2);
      v12 = *(_QWORD *)(a1 + 168);
      v13 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v14 = v12 + 88LL * v13;
      v15 = *(_QWORD *)(v14 + 24);
      if ( v15 == v11 )
        break;
      v33 = 1;
      while ( v15 != -4096 )
      {
        v34 = v33 + 1;
        v13 = (v10 - 1) & (v33 + v13);
        v14 = v12 + 88LL * v13;
        v15 = *(_QWORD *)(v14 + 24);
        if ( v11 == v15 )
          goto LABEL_4;
        v33 = v34;
      }
LABEL_16:
      v9 += 32;
      if ( v36 == v9 )
        goto LABEL_17;
    }
LABEL_4:
    if ( v14 == v12 + 88 * v10 )
      goto LABEL_16;
    v16 = *(_QWORD **)(v14 + 40);
    v17 = &v16[4 * *(unsigned int *)(v14 + 48)];
    if ( v17 == v16 )
      goto LABEL_33;
    v18 = 0;
    v19 = 0;
    do
    {
      v21 = v16[2];
      if ( a2 != v21 )
      {
        v18 |= v21 != 0;
        v20 = v18 & v19;
        goto LABEL_8;
      }
      if ( !a2 )
      {
        v20 = v18;
        v19 = 1;
LABEL_8:
        if ( v20 )
          goto LABEL_16;
        goto LABEL_9;
      }
      if ( a2 != -4096 && a2 != -8192 )
      {
        v35 = v14;
        sub_BD60C0(v16);
        v14 = v35;
      }
      v19 = 1;
      v16[2] = 0;
      if ( v18 )
        goto LABEL_16;
LABEL_9:
      v16 += 4;
    }
    while ( v16 != v17 );
    if ( v18 )
      goto LABEL_16;
    v25 = *(_QWORD **)(v14 + 40);
    v16 = &v25[4 * *(unsigned int *)(v14 + 48)];
    if ( v25 != v16 )
    {
      do
      {
        v26 = *(v16 - 2);
        v16 -= 4;
        LOBYTE(v17) = v26 != -4096;
        if ( ((unsigned __int8)v17 & (v26 != 0)) != 0 && v26 != -8192 )
        {
          v37 = v14;
          sub_BD60C0(v16);
          v14 = v37;
        }
      }
      while ( v25 != v16 );
      v16 = *(_QWORD **)(v14 + 40);
    }
LABEL_33:
    if ( v16 != (_QWORD *)(v14 + 56) )
    {
      v38 = v14;
      _libc_free(v16, v17);
      v14 = v38;
    }
    v41 = -8192;
    v42 = 0;
    v40[0] = 2;
    v27 = *(_QWORD *)(v14 + 24);
    v40[1] = 0;
    if ( v27 == -8192 )
    {
      *(_QWORD *)(v14 + 32) = 0;
    }
    else
    {
      if ( v27 == -4096 || !v27 )
      {
        *(_QWORD *)(v14 + 24) = -8192;
        v31 = v42;
        v32 = v41 != -8192 && v41 != -4096 && v41 != 0;
        goto LABEL_43;
      }
      v28 = (unsigned __int64 *)(v14 + 8);
      v39 = v14;
      sub_BD60C0((_QWORD *)(v14 + 8));
      v29 = v41;
      v30 = v41 == -4096;
      *(_QWORD *)(v39 + 24) = v41;
      if ( v29 == 0 || v30 || v29 == -8192 )
      {
        *(_QWORD *)(v39 + 32) = v42;
      }
      else
      {
        sub_BD6050(v28, v40[0] & 0xFFFFFFFFFFFFFFF8LL);
        v31 = v42;
        v14 = v39;
        v32 = v41 != -8192 && v41 != -4096 && v41 != 0;
LABEL_43:
        *(_QWORD *)(v14 + 32) = v31;
        if ( v32 )
          sub_BD60C0(v40);
      }
    }
    --*(_DWORD *)(a1 + 176);
    v9 += 32;
    ++*(_DWORD *)(a1 + 180);
  }
  while ( v36 != v9 );
LABEL_17:
  result = sub_CFBB40(a1 + 16, a2);
  v23 = v43;
  v24 = &v43[32 * (unsigned int)v44];
  if ( v43 != (_BYTE *)v24 )
  {
    do
    {
      result = *(v24 - 2);
      v24 -= 4;
      if ( result != 0 && result != -4096 && result != -8192 )
        result = sub_BD60C0(v24);
    }
    while ( v23 != v24 );
    v24 = v43;
  }
  if ( v24 != (_QWORD *)v45 )
    return _libc_free(v24, a2);
  return result;
}
