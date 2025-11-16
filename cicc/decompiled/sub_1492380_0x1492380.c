// Function: sub_1492380
// Address: 0x1492380
//
__int64 __fastcall sub_1492380(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  _QWORD *v7; // rax
  __int64 v8; // rcx
  int v10; // edx
  __int64 v11; // rsi
  __int64 v12; // rdi
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // rax
  char *v19; // rdi
  __int64 v21; // rax
  int v22; // edx
  char v23; // al
  __int64 *v24; // rdi
  __int64 v25; // rdi
  unsigned int v26; // esi
  int v27; // eax
  int v28; // eax
  __int64 v29; // rax
  int v30; // eax
  int v31; // r9d
  __int64 v32; // [rsp+8h] [rbp-F8h]
  __int64 *v33; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v34; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v35; // [rsp+28h] [rbp-D8h]
  char *v36; // [rsp+30h] [rbp-D0h]
  __int64 v37; // [rsp+38h] [rbp-C8h]
  char v38; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v39; // [rsp+60h] [rbp-A0h] BYREF
  char *v40[2]; // [rsp+68h] [rbp-98h] BYREF
  _BYTE v41[24]; // [rsp+78h] [rbp-88h] BYREF
  __int64 v42; // [rsp+90h] [rbp-70h] BYREF
  char *v43; // [rsp+98h] [rbp-68h] BYREF
  __int64 v44; // [rsp+A0h] [rbp-60h]
  _BYTE v45[24]; // [rsp+A8h] [rbp-58h] BYREF
  char v46; // [rsp+C0h] [rbp-40h]

  v7 = *(_QWORD **)(a3 + 24);
  if ( *(_BYTE *)(*v7 + 8LL) != 11 )
    goto LABEL_11;
  v8 = *(_QWORD *)(a2 + 64);
  v10 = *(_DWORD *)(v8 + 24);
  if ( !v10 )
    goto LABEL_11;
  v11 = v7[5];
  v12 = *(_QWORD *)(v8 + 8);
  v13 = v10 - 1;
  v14 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v11 != *v15 )
  {
    v30 = 1;
    while ( v16 != -8 )
    {
      v31 = v30 + 1;
      v14 = v13 & (v30 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v11 == *v15 )
        goto LABEL_4;
      v30 = v31;
    }
    goto LABEL_11;
  }
LABEL_4:
  v17 = v15[1];
  if ( !v17 || v11 != **(_QWORD **)(v17 + 32) )
  {
LABEL_11:
    *(_BYTE *)(a1 + 48) = 0;
    return a1;
  }
  v43 = (char *)v15[1];
  v42 = a3;
  v32 = v17;
  if ( !(unsigned __int8)sub_145F9E0(a2 + 1000, &v42, &v39)
    || v39 == (__int64 *)(*(_QWORD *)(a2 + 1008) + ((unsigned __int64)*(unsigned int *)(a2 + 1024) << 6)) )
  {
    sub_1491860((__int64)&v42, a2, a3, a4, a5);
    if ( v46 )
    {
      v18 = v42;
      *(_BYTE *)(a1 + 48) = 1;
      *(_QWORD *)a1 = v18;
      *(_QWORD *)(a1 + 8) = a1 + 24;
      *(_QWORD *)(a1 + 16) = 0x300000000LL;
      if ( !(_DWORD)v44 )
      {
LABEL_9:
        v19 = v43;
        if ( v43 == v45 )
          return a1;
        goto LABEL_20;
      }
      sub_14532C0(a1 + 8, &v43);
LABEL_25:
      if ( !v46 )
        return a1;
      goto LABEL_9;
    }
    v34 = a3;
    v39 = (__int64 *)(a3 + 32);
    v36 = &v38;
    v37 = 0x300000000LL;
    v40[0] = v41;
    v40[1] = (char *)0x300000000LL;
    v35 = v32;
    v23 = sub_145F9E0(a2 + 1000, &v34, &v33);
    v24 = v33;
    if ( v23 )
    {
LABEL_22:
      v25 = (__int64)(v24 + 3);
      *(_QWORD *)(v25 - 8) = v39;
      sub_14532C0(v25, v40);
      if ( v40[0] != v41 )
        _libc_free((unsigned __int64)v40[0]);
      *(_BYTE *)(a1 + 48) = 0;
      goto LABEL_25;
    }
    v26 = *(_DWORD *)(a2 + 1024);
    v27 = *(_DWORD *)(a2 + 1016);
    ++*(_QWORD *)(a2 + 1000);
    v28 = v27 + 1;
    if ( 4 * v28 >= 3 * v26 )
    {
      v26 *= 2;
    }
    else if ( v26 - *(_DWORD *)(a2 + 1020) - v28 > v26 >> 3 )
    {
      goto LABEL_30;
    }
    sub_1468310(a2 + 1000, v26);
    sub_145F9E0(a2 + 1000, &v34, &v33);
    v24 = v33;
    v28 = *(_DWORD *)(a2 + 1016) + 1;
LABEL_30:
    *(_DWORD *)(a2 + 1016) = v28;
    if ( *v24 != -8 || v24[1] != -8 )
      --*(_DWORD *)(a2 + 1020);
    v29 = v34;
    v24[2] = 0;
    *v24 = v29;
    v24[1] = v35;
    v24[3] = (__int64)(v24 + 5);
    v24[4] = 0x300000000LL;
    goto LABEL_22;
  }
  v21 = v39[2];
  v43 = v45;
  v42 = v21;
  v44 = 0x300000000LL;
  if ( *((_DWORD *)v39 + 8) )
  {
    sub_14531E0((__int64)&v43, (__int64)(v39 + 3));
    v21 = v42;
  }
  if ( v21 == a3 + 32 )
  {
    *(_BYTE *)(a1 + 48) = 0;
  }
  else
  {
    *(_QWORD *)a1 = v21;
    v22 = v44;
    *(_QWORD *)(a1 + 8) = a1 + 24;
    *(_BYTE *)(a1 + 48) = 1;
    *(_QWORD *)(a1 + 16) = 0x300000000LL;
    if ( v22 )
      sub_14532C0(a1 + 8, &v43);
  }
  v19 = v43;
  if ( v43 != v45 )
LABEL_20:
    _libc_free((unsigned __int64)v19);
  return a1;
}
