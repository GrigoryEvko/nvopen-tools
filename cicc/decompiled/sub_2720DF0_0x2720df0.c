// Function: sub_2720DF0
// Address: 0x2720df0
//
void __fastcall sub_2720DF0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 *v8; // r12
  char v9; // di
  __int64 *v10; // rbx
  __int64 v11; // rsi
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 v18; // rdx
  _BYTE *v19; // r15
  _BYTE *v20; // rbx
  unsigned __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // rsi
  __int64 *v24[2]; // [rsp-218h] [rbp-218h] BYREF
  char v25; // [rsp-208h] [rbp-208h]
  __int64 *v26; // [rsp-200h] [rbp-200h]
  __int64 v27; // [rsp-1F8h] [rbp-1F8h]
  __int64 v28; // [rsp-1E8h] [rbp-1E8h] BYREF
  __int64 *v29; // [rsp-1E0h] [rbp-1E0h]
  __int64 v30; // [rsp-1D8h] [rbp-1D8h]
  int v31; // [rsp-1D0h] [rbp-1D0h]
  char v32; // [rsp-1CCh] [rbp-1CCh]
  __int64 v33; // [rsp-1C8h] [rbp-1C8h] BYREF
  _BYTE *v34; // [rsp-148h] [rbp-148h] BYREF
  __int64 v35; // [rsp-140h] [rbp-140h]
  _BYTE v36[312]; // [rsp-138h] [rbp-138h] BYREF

  v6 = *(unsigned int *)(a1 + 1472);
  if ( !(_DWORD)v6 )
    return;
  v8 = *(__int64 **)(a1 + 1464);
  v28 = 0;
  v9 = 1;
  v30 = 16;
  v10 = &v8[v6];
  v31 = 0;
  v29 = &v33;
  v32 = 1;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *v8;
        if ( v9 )
          break;
LABEL_27:
        ++v8;
        sub_C8CC70((__int64)&v28, v11, (__int64)a3, a4, a5, a6);
        v9 = v32;
        if ( v10 == v8 )
          goto LABEL_9;
      }
      v12 = v29;
      a4 = HIDWORD(v30);
      a3 = &v29[HIDWORD(v30)];
      if ( v29 != a3 )
        break;
LABEL_29:
      if ( HIDWORD(v30) >= (unsigned int)v30 )
        goto LABEL_27;
      a4 = (unsigned int)(HIDWORD(v30) + 1);
      ++v8;
      ++HIDWORD(v30);
      *a3 = v11;
      v9 = v32;
      ++v28;
      if ( v10 == v8 )
        goto LABEL_9;
    }
    while ( v11 != *v12 )
    {
      if ( a3 == ++v12 )
        goto LABEL_29;
    }
    ++v8;
  }
  while ( v10 != v8 );
LABEL_9:
  v24[1] = 0;
  v35 = 0x2000000000LL;
  v13 = *(__int64 **)(a1 + 16);
  v34 = v36;
  v24[0] = v13;
  v27 = a1 + 1608;
  v26 = &v28;
  v25 = 1;
  sub_271EE20(v24, (__int64)&v34, (__int64)a3, a4, a5, a6);
  ++*(_QWORD *)(a1 + 1608);
  if ( *(_BYTE *)(a1 + 1636) )
    goto LABEL_14;
  v17 = 4 * (*(_DWORD *)(a1 + 1628) - *(_DWORD *)(a1 + 1632));
  v18 = *(unsigned int *)(a1 + 1624);
  if ( v17 < 0x20 )
    v17 = 32;
  if ( (unsigned int)v18 > v17 )
  {
    sub_C8C990(a1 + 1608, (__int64)&v34);
  }
  else
  {
    memset(*(void **)(a1 + 1616), -1, 8 * v18);
LABEL_14:
    *(_QWORD *)(a1 + 1628) = 0;
  }
  v19 = v34;
  v20 = &v34[8 * (unsigned int)v35];
  if ( v20 != v34 )
  {
    do
    {
      v21 = *(_QWORD *)(*(_QWORD *)v19 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v21 == *(_QWORD *)v19 + 48LL )
      {
        v23 = 0;
      }
      else
      {
        if ( !v21 )
          BUG();
        v22 = *(unsigned __int8 *)(v21 - 24);
        v23 = v21 - 24;
        if ( (unsigned int)(v22 - 30) >= 0xB )
          v23 = 0;
      }
      v19 += 8;
      sub_27207D0(a1, v23, v14, v15, v16);
    }
    while ( v20 != v19 );
    v19 = v34;
  }
  if ( v19 != v36 )
    _libc_free((unsigned __int64)v19);
  if ( !v32 )
    _libc_free((unsigned __int64)v29);
}
