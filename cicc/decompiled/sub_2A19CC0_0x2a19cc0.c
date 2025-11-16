// Function: sub_2A19CC0
// Address: 0x2a19cc0
//
void __fastcall sub_2A19CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rbx
  _BYTE *v10; // rax
  int v11; // edx
  _BYTE *v12; // rdi
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r11
  __int64 v18; // r9
  __int64 v19; // r15
  unsigned __int64 v20; // r8
  __int64 v21; // rbx
  _BYTE *v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  _BYTE *v28; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29; // [rsp+28h] [rbp-58h]
  _BYTE v30[80]; // [rsp+30h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v28 = v30;
  v7 = v6 - *(_QWORD *)(a1 + 8);
  v29 = 0x400000000LL;
  v8 = v7 >> 3;
  v9 = v7 >> 3;
  if ( (unsigned __int64)v7 > 0x20 )
  {
    v27 = v6;
    sub_C8D5F0((__int64)&v28, v30, v7 >> 3, 8u, v6, a6);
    v12 = v28;
    v11 = v29;
    v6 = v27;
    v10 = &v28[8 * (unsigned int)v29];
  }
  else
  {
    v10 = v30;
    v11 = 0;
    v12 = v30;
  }
  if ( v7 > 0 )
  {
    do
    {
      v10 += 8;
      *((_QWORD *)v10 - 1) = *(_QWORD *)(v6 - 8 * v8 + 8 * v9-- - 8);
    }
    while ( v9 );
    v11 = v29;
    v12 = v28;
  }
  v13 = v11 + v8;
  for ( LODWORD(v29) = v13; v13; v12 = v28 )
  {
    v14 = v13;
    v15 = v13 - 1;
    v16 = *(_QWORD *)&v12[8 * v14 - 8];
    LODWORD(v29) = v15;
    v17 = *(_QWORD *)(v16 + 16);
    v18 = v17 - *(_QWORD *)(v16 + 8);
    v19 = v18 >> 3;
    v20 = (v18 >> 3) + v15;
    v21 = v18 >> 3;
    if ( v20 > HIDWORD(v29) )
    {
      v25 = *(_QWORD *)(v16 + 16) - *(_QWORD *)(v16 + 8);
      v26 = *(_QWORD *)(v16 + 16);
      sub_C8D5F0((__int64)&v28, v30, v19 + v15, 8u, v20, v18);
      v12 = v28;
      v15 = (unsigned int)v29;
      v18 = v25;
      v17 = v26;
    }
    v22 = &v12[8 * v15];
    if ( v18 > 0 )
    {
      do
      {
        v22 += 8;
        *((_QWORD *)v22 - 1) = *(_QWORD *)(v17 - 8 * v19 + 8 * v21-- - 8);
      }
      while ( v21 );
      LODWORD(v15) = v29;
    }
    v23 = *(unsigned int *)(a2 + 8);
    v24 = *(unsigned int *)(a2 + 12);
    LODWORD(v29) = v19 + v15;
    if ( v23 + 1 > v24 )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v23 + 1, 8u, v20, v18);
      v23 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v23) = v16;
    v13 = v29;
    ++*(_DWORD *)(a2 + 8);
  }
  if ( v12 != v30 )
    _libc_free((unsigned __int64)v12);
}
