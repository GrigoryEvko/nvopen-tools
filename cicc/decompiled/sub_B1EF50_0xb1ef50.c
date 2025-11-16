// Function: sub_B1EF50
// Address: 0xb1ef50
//
__int64 __fastcall sub_B1EF50(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // r14
  __int64 v6; // rsi
  _QWORD *v7; // r14
  __int64 v8; // r12
  __int64 v9; // rsi
  int i; // eax
  char *v11; // rdx
  __int64 v12; // r15
  int v13; // r10d
  __int64 v14; // rax
  int v15; // r10d
  __int64 v16; // r13
  __int64 v17; // rax
  int v18; // edx
  char *v19; // r10
  __int64 v20; // rdx
  __int64 v21; // r11
  __int64 *v22; // r13
  _QWORD *v23; // rax
  __int64 v24; // r14
  char *v25; // rdx
  __int64 *v26; // rax
  _BYTE *v27; // rbx
  __int64 result; // rax
  _BYTE *v29; // r12
  _BYTE *v30; // rdi
  _QWORD *v31; // [rsp+18h] [rbp-1508h]
  __int64 v32; // [rsp+20h] [rbp-1500h]
  unsigned int v35; // [rsp+54h] [rbp-14CCh]
  int v36; // [rsp+58h] [rbp-14C8h]
  char *v37; // [rsp+58h] [rbp-14C8h]
  __int64 *v38; // [rsp+60h] [rbp-14C0h] BYREF
  int v39; // [rsp+68h] [rbp-14B8h]
  char v40; // [rsp+70h] [rbp-14B0h] BYREF
  char *v41; // [rsp+B0h] [rbp-1470h] BYREF
  __int64 v42; // [rsp+B8h] [rbp-1468h]
  _BYTE v43[1024]; // [rsp+C0h] [rbp-1460h] BYREF
  _QWORD v44[2]; // [rsp+4C0h] [rbp-1060h] BYREF
  _QWORD v45[64]; // [rsp+4D0h] [rbp-1050h] BYREF
  _BYTE *v46; // [rsp+6D0h] [rbp-E50h]
  __int64 v47; // [rsp+6D8h] [rbp-E48h]
  _BYTE v48[3584]; // [rsp+6E0h] [rbp-E40h] BYREF
  __int64 v49; // [rsp+14E0h] [rbp-40h]

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 104);
  sub_B1AD90((__int64 *)(a1 + 24), a2);
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 116) = 0;
  *(_QWORD *)(a1 + 104) = v4;
  if ( a2 )
  {
    v3 = *(_QWORD *)(a2 + 16);
    if ( v3 )
    {
      v5 = *(_QWORD *)(a2 + 8);
      if ( v3 != v5 )
        sub_B1BF30(*(_QWORD *)(a2 + 8), *(_QWORD *)(a2 + 16));
      if ( v3 + 304 != v5 + 304 )
        sub_B1BF30(v5 + 304, v3 + 304);
      *(_BYTE *)(v5 + 608) = *(_BYTE *)(v3 + 608);
      sub_B18600(v5 + 616, v3 + 616);
      v3 = a2;
      v4 = *(_QWORD *)(a1 + 104);
    }
  }
  v49 = v3;
  v44[0] = v45;
  v44[1] = 0x4000000001LL;
  v46 = v48;
  v47 = 0x4000000000LL;
  v42 = 0x100000000LL;
  v41 = v43;
  v6 = *(_QWORD *)(v4 + 80);
  v45[0] = 0;
  if ( v6 )
    v6 -= 24;
  sub_B1A4E0((__int64)&v41, v6);
  sub_B187A0(a1, &v41);
  if ( v41 != v43 )
    _libc_free(v41, &v41);
  v7 = v44;
  v8 = **(_QWORD **)a1;
  v39 = 0;
  v38 = (__int64 *)v8;
  sub_B1C510(&v41, &v38, 1);
  v9 = v8;
  v35 = 0;
  *(_DWORD *)(sub_B1E0B0((__int64)v44, v8) + 4) = 0;
  for ( i = v42; (_DWORD)v42; i = v42 )
  {
    while ( 1 )
    {
      v11 = &v41[16 * i - 16];
      v12 = *(_QWORD *)v11;
      v13 = *((_DWORD *)v11 + 2);
      LODWORD(v42) = i - 1;
      v9 = v12;
      v36 = v13;
      v14 = sub_B1E0B0((__int64)v7, v12);
      v15 = v36;
      v16 = v14;
      v17 = *(unsigned int *)(v14 + 32);
      if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 36) )
      {
        v9 = v16 + 40;
        sub_C8D5F0(v16 + 24, v16 + 40, v17 + 1, 4);
        v17 = *(unsigned int *)(v16 + 32);
        v15 = v36;
      }
      *(_DWORD *)(*(_QWORD *)(v16 + 24) + 4 * v17) = v15;
      v18 = *(_DWORD *)v16;
      ++*(_DWORD *)(v16 + 32);
      if ( !v18 )
      {
        ++v35;
        *(_DWORD *)(v16 + 4) = v15;
        *(_DWORD *)(v16 + 12) = v35;
        *(_DWORD *)(v16 + 8) = v35;
        *(_DWORD *)v16 = v35;
        sub_B1A4E0((__int64)v7, v12);
        v9 = v12;
        sub_B1D150(&v38, v12, v49);
        v19 = (char *)&v38[v39];
        if ( v38 != (__int64 *)v19 )
        {
          v20 = (unsigned int)v42;
          v21 = v35;
          v22 = v38;
          v23 = v7;
          do
          {
            v24 = *v22;
            v2 = v21 | v2 & 0xFFFFFFFF00000000LL;
            if ( v20 + 1 > (unsigned __int64)HIDWORD(v42) )
            {
              v31 = v23;
              v32 = v21;
              v37 = v19;
              sub_C8D5F0(&v41, v43, v20 + 1, 16);
              v20 = (unsigned int)v42;
              v23 = v31;
              v21 = v32;
              v19 = v37;
            }
            v25 = &v41[16 * v20];
            ++v22;
            *(_QWORD *)v25 = v24;
            *((_QWORD *)v25 + 1) = v2;
            v9 = (unsigned int)v42;
            v20 = (unsigned int)(v42 + 1);
            LODWORD(v42) = v42 + 1;
          }
          while ( v19 != (char *)v22 );
          v19 = (char *)v38;
          v7 = v23;
        }
        if ( v19 != &v40 )
          break;
      }
      i = v42;
      if ( !(_DWORD)v42 )
        goto LABEL_26;
    }
    _libc_free(v19, v9);
  }
LABEL_26:
  if ( v41 != v43 )
    _libc_free(v41, v9);
  sub_B1E260((__int64)v7);
  if ( a2 )
    *(_BYTE *)a2 = 1;
  if ( !*(_DWORD *)(a1 + 8) )
    return sub_B1AC50((__int64)v7, v9);
  v26 = (__int64 *)sub_B1B5D0(a1, **(_QWORD **)a1, 0);
  *(_QWORD *)(a1 + 96) = v26;
  sub_B1E720(v7, a1, *v26);
  v27 = v46;
  result = 7LL * (unsigned int)v47;
  v29 = &v46[56 * (unsigned int)v47];
  if ( v46 != v29 )
  {
    do
    {
      v29 -= 56;
      v30 = (_BYTE *)*((_QWORD *)v29 + 3);
      result = (__int64)(v29 + 40);
      if ( v30 != v29 + 40 )
        result = _libc_free(v30, a1);
    }
    while ( v27 != v29 );
    v29 = v46;
  }
  if ( v29 != v48 )
    result = _libc_free(v29, a1);
  if ( (_QWORD *)v44[0] != v45 )
    return _libc_free(v44[0], a1);
  return result;
}
