// Function: sub_2EA5BA0
// Address: 0x2ea5ba0
//
__int64 __fastcall sub_2EA5BA0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // r9
  __int64 v5; // r8
  __int64 v6; // r10
  __int64 v7; // rax
  _BYTE *v8; // rdx
  int v9; // ecx
  _BYTE *v10; // rdi
  int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // r13
  _BYTE *v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // r8d
  __int64 v22; // rax
  void *v23; // r10
  unsigned __int64 v24; // rdx
  __int64 v25; // r9
  _QWORD *v26; // rdi
  const void *v28; // [rsp+0h] [rbp-110h]
  __int64 v29; // [rsp+8h] [rbp-108h]
  int v30; // [rsp+10h] [rbp-100h]
  __int64 v31; // [rsp+10h] [rbp-100h]
  __int64 v32; // [rsp+20h] [rbp-F0h]
  __int64 v33; // [rsp+28h] [rbp-E8h]
  __int64 v34; // [rsp+28h] [rbp-E8h]
  __int64 v35; // [rsp+28h] [rbp-E8h]
  __int64 i; // [rsp+40h] [rbp-D0h]
  __int64 v37; // [rsp+48h] [rbp-C8h]
  int v38; // [rsp+48h] [rbp-C8h]
  void *v39; // [rsp+48h] [rbp-C8h]
  __int64 v40; // [rsp+48h] [rbp-C8h]
  void *src; // [rsp+80h] [rbp-90h] BYREF
  __int64 v42; // [rsp+88h] [rbp-88h]
  _QWORD v43[4]; // [rsp+90h] [rbp-80h] BYREF
  _BYTE *v44; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v45; // [rsp+B8h] [rbp-58h]
  _BYTE v46[80]; // [rsp+C0h] [rbp-50h] BYREF

  v2 = (__int64)a1;
  *a1 = a1 + 2;
  v28 = a1 + 2;
  a1[1] = 0x400000000LL;
  v32 = *(_QWORD *)(a2 + 32);
  for ( i = *(_QWORD *)(a2 + 40); v32 != i; i -= 8 )
  {
    v3 = *(_QWORD *)(i - 8);
    src = v43;
    v42 = 0x400000001LL;
    v43[0] = v3;
    v4 = *(_QWORD *)(v3 + 16);
    v45 = 0x400000000LL;
    v5 = v4 - *(_QWORD *)(v3 + 8);
    v44 = v46;
    v6 = v5 >> 3;
    v7 = v5 >> 3;
    if ( (unsigned __int64)v5 > 0x20 )
    {
      v29 = v5;
      v31 = v5 >> 3;
      v35 = v4;
      v40 = v5 >> 3;
      sub_C8D5F0((__int64)&v44, v46, v5 >> 3, 8u, v5, v4);
      v10 = v44;
      v9 = v45;
      v6 = v40;
      v4 = v35;
      v7 = v31;
      v5 = v29;
      v8 = &v44[8 * (unsigned int)v45];
    }
    else
    {
      v8 = v46;
      v9 = 0;
      v10 = v46;
    }
    if ( v5 > 0 )
    {
      do
      {
        v8 += 8;
        *((_QWORD *)v8 - 1) = *(_QWORD *)(v4 - 8 * v6 + 8 * v7-- - 8);
      }
      while ( v7 );
      v9 = v45;
      v10 = v44;
    }
    v11 = v9 + v6;
    LODWORD(v45) = v11;
    if ( v11 )
    {
      v33 = v2;
      while ( 1 )
      {
        v12 = (unsigned int)(v11 - 1);
        v13 = *(_QWORD *)&v10[8 * v11 - 8];
        LODWORD(v45) = v11 - 1;
        v14 = *(_QWORD *)(v13 + 16);
        v15 = v14 - *(_QWORD *)(v13 + 8);
        v16 = v15 >> 3;
        v17 = v15 >> 3;
        if ( (v15 >> 3) + v12 > (unsigned __int64)HIDWORD(v45) )
        {
          v37 = *(_QWORD *)(v13 + 16) - *(_QWORD *)(v13 + 8);
          sub_C8D5F0((__int64)&v44, v46, v16 + v12, 8u, v5, v15);
          v10 = v44;
          v12 = (unsigned int)v45;
          v15 = v37;
        }
        v18 = &v10[8 * v12];
        if ( v15 > 0 )
        {
          do
          {
            v18 += 8;
            *((_QWORD *)v18 - 1) = *(_QWORD *)(v14 - 8 * v16 + 8 * v17-- - 8);
          }
          while ( v17 );
          LODWORD(v12) = v45;
        }
        v19 = (unsigned int)v42;
        LODWORD(v45) = v16 + v12;
        v20 = (unsigned int)v42 + 1LL;
        if ( v20 > HIDWORD(v42) )
        {
          sub_C8D5F0((__int64)&src, v43, v20, 8u, v5, v15);
          v19 = (unsigned int)v42;
        }
        *((_QWORD *)src + v19) = v13;
        v11 = v45;
        LODWORD(v42) = v42 + 1;
        if ( !(_DWORD)v45 )
          break;
        v10 = v44;
      }
      v2 = v33;
      v10 = v44;
    }
    if ( v10 != v46 )
      _libc_free((unsigned __int64)v10);
    v21 = v42;
    v22 = *(unsigned int *)(v2 + 8);
    v23 = src;
    v24 = (unsigned int)v42 + v22;
    v25 = 8LL * (unsigned int)v42;
    if ( v24 > *(unsigned int *)(v2 + 12) )
    {
      v30 = v42;
      v34 = 8LL * (unsigned int)v42;
      v39 = src;
      sub_C8D5F0(v2, v28, v24, 8u, (unsigned int)v42, v25);
      v22 = *(unsigned int *)(v2 + 8);
      v21 = v30;
      v25 = v34;
      v23 = v39;
    }
    if ( v25 )
    {
      v38 = v21;
      memcpy((void *)(*(_QWORD *)v2 + 8 * v22), v23, v25);
      LODWORD(v22) = *(_DWORD *)(v2 + 8);
      v21 = v38;
    }
    v26 = src;
    *(_DWORD *)(v2 + 8) = v22 + v21;
    if ( v26 != v43 )
      _libc_free((unsigned __int64)v26);
  }
  return v2;
}
