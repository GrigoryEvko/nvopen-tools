// Function: sub_D47CF0
// Address: 0xd47cf0
//
__int64 __fastcall sub_D47CF0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v7; // r10
  __int64 v8; // rax
  _BYTE *v9; // rdx
  int v10; // ecx
  _BYTE *v11; // rdi
  int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // r9
  __int64 v17; // r15
  __int64 v18; // r13
  _BYTE *v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  int v22; // r8d
  __int64 v23; // rax
  void *v24; // r10
  unsigned __int64 v25; // rdx
  __int64 v26; // r9
  _QWORD *v27; // rdi
  const void *v29; // [rsp+0h] [rbp-110h]
  __int64 v30; // [rsp+8h] [rbp-108h]
  int v31; // [rsp+10h] [rbp-100h]
  __int64 v32; // [rsp+10h] [rbp-100h]
  __int64 v33; // [rsp+20h] [rbp-F0h]
  __int64 v34; // [rsp+28h] [rbp-E8h]
  __int64 v35; // [rsp+28h] [rbp-E8h]
  __int64 v36; // [rsp+28h] [rbp-E8h]
  __int64 i; // [rsp+40h] [rbp-D0h]
  __int64 v38; // [rsp+48h] [rbp-C8h]
  int v39; // [rsp+48h] [rbp-C8h]
  void *v40; // [rsp+48h] [rbp-C8h]
  __int64 v41; // [rsp+48h] [rbp-C8h]
  void *src; // [rsp+80h] [rbp-90h] BYREF
  __int64 v43; // [rsp+88h] [rbp-88h]
  _QWORD v44[4]; // [rsp+90h] [rbp-80h] BYREF
  _BYTE *v45; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v46; // [rsp+B8h] [rbp-58h]
  _BYTE v47[80]; // [rsp+C0h] [rbp-50h] BYREF

  v2 = (__int64)a1;
  *a1 = a1 + 2;
  v29 = a1 + 2;
  a1[1] = 0x400000000LL;
  v33 = *(_QWORD *)(a2 + 32);
  for ( i = *(_QWORD *)(a2 + 40); v33 != i; i -= 8 )
  {
    v3 = *(_QWORD *)(i - 8);
    src = v44;
    v43 = 0x400000001LL;
    v4 = 0x400000000LL;
    v44[0] = v3;
    v5 = *(_QWORD *)(v3 + 16);
    v46 = 0x400000000LL;
    v6 = v5 - *(_QWORD *)(v3 + 8);
    v45 = v47;
    v7 = v6 >> 3;
    v8 = v6 >> 3;
    if ( (unsigned __int64)v6 > 0x20 )
    {
      v4 = (__int64)v47;
      v30 = v6;
      v32 = v6 >> 3;
      v36 = v5;
      v41 = v6 >> 3;
      sub_C8D5F0((__int64)&v45, v47, v6 >> 3, 8u, v6, v5);
      v11 = v45;
      v10 = v46;
      v7 = v41;
      v5 = v36;
      v8 = v32;
      v6 = v30;
      v9 = &v45[8 * (unsigned int)v46];
    }
    else
    {
      v9 = v47;
      v10 = 0;
      v11 = v47;
    }
    if ( v6 > 0 )
    {
      v4 = v5 - 8 * v7;
      do
      {
        v9 += 8;
        *((_QWORD *)v9 - 1) = *(_QWORD *)(v4 + 8 * v8-- - 8);
      }
      while ( v8 );
      v10 = v46;
      v11 = v45;
    }
    v12 = v10 + v7;
    LODWORD(v46) = v12;
    if ( v12 )
    {
      v34 = v2;
      while ( 1 )
      {
        v13 = (unsigned int)(v12 - 1);
        v14 = *(_QWORD *)&v11[8 * v12 - 8];
        LODWORD(v46) = v12 - 1;
        v15 = *(_QWORD *)(v14 + 16);
        v16 = v15 - *(_QWORD *)(v14 + 8);
        v17 = v16 >> 3;
        v18 = v16 >> 3;
        if ( (v16 >> 3) + v13 > (unsigned __int64)HIDWORD(v46) )
        {
          v4 = (__int64)v47;
          v38 = *(_QWORD *)(v14 + 16) - *(_QWORD *)(v14 + 8);
          sub_C8D5F0((__int64)&v45, v47, v17 + v13, 8u, v6, v16);
          v11 = v45;
          v13 = (unsigned int)v46;
          v16 = v38;
        }
        v19 = &v11[8 * v13];
        if ( v16 > 0 )
        {
          do
          {
            v19 += 8;
            *((_QWORD *)v19 - 1) = *(_QWORD *)(v15 - 8 * v17 + 8 * v18-- - 8);
          }
          while ( v18 );
          LODWORD(v13) = v46;
        }
        v20 = (unsigned int)v43;
        LODWORD(v46) = v17 + v13;
        v21 = (unsigned int)v43 + 1LL;
        if ( v21 > HIDWORD(v43) )
        {
          v4 = (__int64)v44;
          sub_C8D5F0((__int64)&src, v44, v21, 8u, v6, v16);
          v20 = (unsigned int)v43;
        }
        *((_QWORD *)src + v20) = v14;
        v12 = v46;
        LODWORD(v43) = v43 + 1;
        if ( !(_DWORD)v46 )
          break;
        v11 = v45;
      }
      v2 = v34;
      v11 = v45;
    }
    if ( v11 != v47 )
      _libc_free(v11, v4);
    v22 = v43;
    v23 = *(unsigned int *)(v2 + 8);
    v24 = src;
    v25 = (unsigned int)v43 + v23;
    v26 = 8LL * (unsigned int)v43;
    if ( v25 > *(unsigned int *)(v2 + 12) )
    {
      v4 = (__int64)v29;
      v31 = v43;
      v35 = 8LL * (unsigned int)v43;
      v40 = src;
      sub_C8D5F0(v2, v29, v25, 8u, (unsigned int)v43, v26);
      v23 = *(unsigned int *)(v2 + 8);
      v22 = v31;
      v26 = v35;
      v24 = v40;
    }
    if ( v26 )
    {
      v4 = (__int64)v24;
      v39 = v22;
      memcpy((void *)(*(_QWORD *)v2 + 8 * v23), v24, v26);
      LODWORD(v23) = *(_DWORD *)(v2 + 8);
      v22 = v39;
    }
    v27 = src;
    *(_DWORD *)(v2 + 8) = v23 + v22;
    if ( v27 != v44 )
      _libc_free(v27, v4);
  }
  return v2;
}
