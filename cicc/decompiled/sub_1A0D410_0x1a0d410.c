// Function: sub_1A0D410
// Address: 0x1a0d410
//
void __fastcall sub_1A0D410(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 *v8; // r8
  int v9; // ecx
  unsigned __int64 v10; // r9
  __int64 *v11; // rax
  int v12; // eax
  int v13; // ecx
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdi
  int v18; // eax
  int v19; // edx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 *v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rax
  int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 *v29; // rcx
  __int64 v30; // rsi
  _QWORD *v31; // r14
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 *v34; // r13
  __int64 v35; // rax
  int v36; // ecx
  int v37; // r8d
  int v38; // eax
  int v39; // r8d
  int v40; // ecx
  int v41; // r8d
  __int64 *v42; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v43; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v45[3]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD *v46; // [rsp+38h] [rbp-A8h]
  __int64 v47[4]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v48; // [rsp+60h] [rbp-80h] BYREF
  __int64 v49; // [rsp+68h] [rbp-78h]
  __int64 v50; // [rsp+70h] [rbp-70h]
  _QWORD *v51; // [rsp+78h] [rbp-68h]
  __int64 *v52; // [rsp+80h] [rbp-60h] BYREF
  __int64 v53; // [rsp+88h] [rbp-58h]
  _BYTE v54[80]; // [rsp+90h] [rbp-50h] BYREF

  v6 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v7 = *(__int64 **)(a2 - 8);
    v8 = &v7[(unsigned __int64)v6 / 8];
  }
  else
  {
    v8 = (__int64 *)a2;
    v7 = (__int64 *)(a2 - v6);
  }
  v9 = 0;
  v52 = (__int64 *)v54;
  v53 = 0x400000000LL;
  v10 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  v11 = (__int64 *)v54;
  if ( (unsigned __int64)v6 > 0x60 )
  {
    v42 = v8;
    v43 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
    sub_16CD150((__int64)&v52, v54, v43, 8, (int)v8, v10);
    v9 = v53;
    v8 = v42;
    LODWORD(v10) = v43;
    v11 = &v52[(unsigned int)v53];
  }
  if ( v7 != v8 )
  {
    do
    {
      if ( v11 )
        *v11 = *v7;
      v7 += 3;
      ++v11;
    }
    while ( v7 != v8 );
    v9 = v53;
  }
  v12 = *(_DWORD *)(a1 + 56);
  LODWORD(v53) = v9 + v10;
  if ( v12 )
  {
    v13 = v12 - 1;
    v14 = *(_QWORD *)(a1 + 40);
    v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( a2 == *v16 )
    {
LABEL_12:
      *v16 = -16;
      --*(_DWORD *)(a1 + 48);
      ++*(_DWORD *)(a1 + 52);
    }
    else
    {
      v38 = 1;
      while ( v17 != -8 )
      {
        v39 = v38 + 1;
        v15 = v13 & (v38 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( a2 == *v16 )
          goto LABEL_12;
        v38 = v39;
      }
    }
  }
  v18 = *(_DWORD *)(a3 + 24);
  v44 = a2;
  if ( v18 )
  {
    v19 = v18 - 1;
    v20 = *(_QWORD *)(a3 + 8);
    v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v22 = (__int64 *)(v20 + 8LL * v21);
    v23 = *v22;
    if ( a2 == *v22 )
    {
LABEL_15:
      *v22 = -16;
      --*(_DWORD *)(a3 + 16);
      ++*(_DWORD *)(a3 + 20);
      sub_1A02020(v45, (_QWORD *)(a3 + 32), &v44);
      v48 = v45[0];
      v24 = *v46;
      v51 = v46;
      v49 = v24;
      v50 = v24 + 512;
      sub_1A0CA40(v47, (_QWORD *)(a3 + 32), &v48);
    }
    else
    {
      v40 = 1;
      while ( v23 != -8 )
      {
        v41 = v40 + 1;
        v21 = v19 & (v40 + v21);
        v22 = (__int64 *)(v20 + 8LL * v21);
        v23 = *v22;
        if ( a2 == *v22 )
          goto LABEL_15;
        v40 = v41;
      }
    }
  }
  v25 = *(_DWORD *)(a1 + 88);
  v44 = a2;
  if ( v25 )
  {
    v26 = v25 - 1;
    v27 = *(_QWORD *)(a1 + 72);
    v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v29 = (__int64 *)(v27 + 8LL * v28);
    v30 = *v29;
    if ( a2 == *v29 )
    {
LABEL_18:
      *v29 = -16;
      --*(_DWORD *)(a1 + 80);
      ++*(_DWORD *)(a1 + 84);
      v31 = (_QWORD *)(a1 + 96);
      sub_1A02020(v45, v31, &v44);
      v48 = v45[0];
      v32 = *v46;
      v51 = v46;
      v49 = v32;
      v50 = v32 + 512;
      sub_1A0CA40(v47, v31, &v48);
    }
    else
    {
      v36 = 1;
      while ( v30 != -8 )
      {
        v37 = v36 + 1;
        v28 = v26 & (v36 + v28);
        v29 = (__int64 *)(v27 + 8LL * v28);
        v30 = *v29;
        if ( a2 == *v29 )
          goto LABEL_18;
        v36 = v37;
      }
    }
  }
  sub_15F20C0((_QWORD *)a2);
  v33 = v52;
  v34 = &v52[(unsigned int)v53];
  if ( v52 != v34 )
  {
    do
    {
      while ( 1 )
      {
        v35 = *v33;
        if ( *(_BYTE *)(*v33 + 16) > 0x17u && !*(_QWORD *)(v35 + 8) )
          break;
        if ( v34 == ++v33 )
          goto LABEL_25;
      }
      ++v33;
      v48 = v35;
      sub_1A062A0(a3, &v48);
    }
    while ( v34 != v33 );
LABEL_25:
    v34 = v52;
  }
  if ( v34 != (__int64 *)v54 )
    _libc_free((unsigned __int64)v34);
}
