// Function: sub_2E0A7D0
// Address: 0x2e0a7d0
//
void __fastcall sub_2E0A7D0(int a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, unsigned int a6)
{
  __int64 *v6; // r15
  __int64 v7; // r9
  int v8; // r11d
  __int64 v10; // r8
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 i; // rbx
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 *v21; // rbx
  __int64 *v22; // r12
  __int64 v23; // rsi
  __int64 *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // [rsp+0h] [rbp-D0h]
  _QWORD *v30; // [rsp+0h] [rbp-D0h]
  __int64 v31; // [rsp+8h] [rbp-C8h]
  int v32; // [rsp+8h] [rbp-C8h]
  __int64 v33; // [rsp+10h] [rbp-C0h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  _QWORD *v35; // [rsp+18h] [rbp-B8h]
  __int64 v36; // [rsp+18h] [rbp-B8h]
  __int64 *v41; // [rsp+50h] [rbp-80h] BYREF
  __int64 v42; // [rsp+58h] [rbp-78h]
  _BYTE v43[112]; // [rsp+60h] [rbp-70h] BYREF

  if ( a1 >= 0 )
    return;
  v6 = *(__int64 **)(a2 + 64);
  v41 = (__int64 *)v43;
  v42 = 0x800000000LL;
  v7 = (__int64)&v6[*(unsigned int *)(a2 + 72)];
  if ( v6 == (__int64 *)v7 )
    return;
  v8 = a1;
  do
  {
    while ( 1 )
    {
      v10 = *v6;
      v11 = *(_QWORD *)(*v6 + 8);
      if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v11 & 6) != 0 )
        break;
LABEL_4:
      if ( (__int64 *)v7 == ++v6 )
        goto LABEL_27;
    }
    v12 = *(_QWORD *)((*(_QWORD *)(*v6 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
    for ( i = v12; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v14 = *(_QWORD *)(v12 + 24) + 48LL;
    while ( 1 )
    {
      v15 = *(_QWORD *)(i + 32);
      v16 = v15 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
      if ( v15 != v16 )
        break;
      i = *(_QWORD *)(i + 8);
      if ( v14 == i )
        break;
      if ( (*(_BYTE *)(i + 44) & 4) == 0 )
      {
        i = *(_QWORD *)(v12 + 24) + 48LL;
        break;
      }
    }
    if ( v16 != v15 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v15 && (*(_BYTE *)(v15 + 3) & 0x10) != 0 && v8 == *(_DWORD *)(v15 + 8) )
        {
          v24 = (__int64 *)(a5[34] + 16LL * ((*(_DWORD *)v15 >> 8) & 0xFFF));
          v25 = *v24;
          v26 = v24[1];
          if ( a6 )
          {
            v29 = v8;
            v31 = v10;
            v33 = v7;
            v35 = a5;
            v27 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64, __int64))(*a5 + 312LL))(a5, a6, v25, v26);
            v8 = v29;
            v10 = v31;
            v7 = v33;
            a5 = v35;
            v25 = v27;
            v26 = v28;
          }
          if ( a4 & v26 | a3 & v25 )
            goto LABEL_4;
        }
        v17 = v15 + 40;
        v18 = v16;
        if ( v17 == v16 )
        {
          while ( 1 )
          {
            i = *(_QWORD *)(i + 8);
            if ( v14 == i )
            {
              v15 = v16;
              v16 = v18;
              goto LABEL_23;
            }
            if ( (*(_BYTE *)(i + 44) & 4) == 0 )
              break;
            v16 = *(_QWORD *)(i + 32);
            v18 = v16 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
            if ( v16 != v18 )
              goto LABEL_33;
          }
          v15 = v16;
          i = v14;
          v16 = v18;
LABEL_23:
          if ( v16 == v15 )
            break;
        }
        else
        {
          v16 = v17;
LABEL_33:
          v15 = v16;
          v16 = v18;
        }
      }
    }
    v19 = (unsigned int)v42;
    v20 = (unsigned int)v42 + 1LL;
    if ( v20 > HIDWORD(v42) )
    {
      v30 = a5;
      v32 = v8;
      v34 = v7;
      v36 = v10;
      sub_C8D5F0((__int64)&v41, v43, v20, 8u, v10, v7);
      v19 = (unsigned int)v42;
      a5 = v30;
      v8 = v32;
      v7 = v34;
      v10 = v36;
    }
    ++v6;
    v41[v19] = v10;
    LODWORD(v42) = v42 + 1;
  }
  while ( (__int64 *)v7 != v6 );
LABEL_27:
  v21 = v41;
  v22 = &v41[(unsigned int)v42];
  if ( v41 != v22 )
  {
    do
    {
      v23 = *v21++;
      sub_2E0A600(a2, v23);
    }
    while ( v22 != v21 );
    v22 = v41;
  }
  if ( v22 != (__int64 *)v43 )
    _libc_free((unsigned __int64)v22);
}
