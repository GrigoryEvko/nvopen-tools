// Function: sub_15A5DE0
// Address: 0x15a5de0
//
void __fastcall sub_15A5DE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // r8d
  __int64 v8; // r9
  __int64 *v9; // rdx
  __int64 v10; // rsi
  _QWORD *v11; // r14
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  int v14; // ecx
  __int64 v15; // r8
  _QWORD *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // edi
  __int64 *v21; // rcx
  __int64 v22; // r9
  unsigned __int64 v23; // r8
  _QWORD *v24; // rbx
  __int64 v25; // r14
  __int64 v26; // rsi
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rdi
  int v34; // ecx
  int v35; // r10d
  __int64 v36; // [rsp+0h] [rbp-D0h]
  int v37; // [rsp+8h] [rbp-C8h]
  int v38; // [rsp+8h] [rbp-C8h]
  _BYTE *v39; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+18h] [rbp-B8h]
  _BYTE v41[176]; // [rsp+20h] [rbp-B0h] BYREF

  v2 = *(_QWORD *)(a2 + 8 * (7LL - *(unsigned int *)(a2 + 8)));
  if ( !v2 || *(_BYTE *)(v2 + 1) != 2 )
    return;
  v40 = 0x1000000000LL;
  v5 = *(unsigned int *)(a1 + 424);
  v39 = v41;
  if ( !(_DWORD)v5 )
    goto LABEL_32;
  v6 = *(_QWORD *)(a1 + 408);
  v7 = 1;
  LODWORD(v8) = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v6 + 32LL * (unsigned int)v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    while ( v10 != -8 )
    {
      v8 = ((_DWORD)v5 - 1) & (unsigned int)(v8 + v7);
      v9 = (__int64 *)(v6 + 32 * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_6;
      ++v7;
    }
    goto LABEL_32;
  }
LABEL_6:
  if ( v9 == (__int64 *)(v6 + 32 * v5) )
  {
LABEL_32:
    v17 = 0;
    goto LABEL_16;
  }
  v11 = (_QWORD *)v9[1];
  v12 = *((unsigned int *)v9 + 4);
  v13 = v41;
  v14 = 0;
  v15 = 8 * v12;
  if ( v12 > 0x10 )
  {
    v36 = 8 * v12;
    v38 = v12;
    sub_16CD150(&v39, v41, v12, 8);
    v14 = v40;
    v15 = v36;
    LODWORD(v12) = v38;
    v13 = &v39[8 * (unsigned int)v40];
  }
  if ( v15 )
  {
    v16 = (_QWORD *)((char *)v13 + v15);
    do
    {
      if ( v13 )
        *v13 = *v11;
      ++v13;
      ++v11;
    }
    while ( v16 != v13 );
    v14 = v40;
  }
  LODWORD(v40) = v14 + v12;
  v17 = (unsigned int)(v14 + v12);
LABEL_16:
  v18 = *(unsigned int *)(a1 + 456);
  if ( (_DWORD)v18 )
  {
    v19 = *(_QWORD *)(a1 + 440);
    v20 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v21 = (__int64 *)(v19 + 32LL * v20);
    v22 = *v21;
    if ( a2 == *v21 )
    {
LABEL_18:
      if ( v21 != (__int64 *)(v19 + 32 * v18) )
      {
        v23 = *((unsigned int *)v21 + 4);
        v24 = (_QWORD *)v21[1];
        v25 = v23;
        if ( v23 > (unsigned __int64)HIDWORD(v40) - v17 )
        {
          v37 = *((_DWORD *)v21 + 4);
          sub_16CD150(&v39, v41, v23 + v17, 8);
          v17 = (unsigned int)v40;
          LODWORD(v23) = v37;
        }
        v26 = (__int64)v39;
        v27 = &v39[8 * v17];
        if ( v25 * 8 )
        {
          v28 = &v27[v25];
          do
          {
            if ( v27 )
              *v27 = *v24;
            ++v27;
            ++v24;
          }
          while ( v28 != v27 );
          LODWORD(v17) = v40;
          v26 = (__int64)v39;
        }
        LODWORD(v40) = v23 + v17;
        v17 = (unsigned int)(v23 + v17);
        goto LABEL_28;
      }
    }
    else
    {
      v34 = 1;
      while ( v22 != -8 )
      {
        v35 = v34 + 1;
        v20 = (v18 - 1) & (v20 + v34);
        v21 = (__int64 *)(v19 + 32LL * v20);
        v22 = *v21;
        if ( a2 == *v21 )
          goto LABEL_18;
        v34 = v35;
      }
    }
  }
  v26 = (__int64)v39;
LABEL_28:
  v29 = sub_15A5DC0(a1, v26, v17);
  v33 = *(_QWORD *)(v2 + 16);
  if ( (v33 & 4) != 0 )
  {
    v26 = v29;
    sub_16302D0(v33 & 0xFFFFFFFFFFFFFFF8LL, v29);
  }
  sub_16307F0(v2, v26, v30, v31, v32);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
}
