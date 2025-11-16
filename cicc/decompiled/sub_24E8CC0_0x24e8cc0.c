// Function: sub_24E8CC0
// Address: 0x24e8cc0
//
__int64 __fastcall sub_24E8CC0(_QWORD *a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rdx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 result; // rax
  __int64 *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 i; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rdx
  char v23; // al
  __int64 *v24; // [rsp+8h] [rbp-E8h]
  int v25; // [rsp+1Ch] [rbp-D4h] BYREF
  unsigned __int64 v26[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+30h] [rbp-C0h]
  _BYTE v28[32]; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v29; // [rsp+60h] [rbp-90h]
  __int64 *v30; // [rsp+70h] [rbp-80h] BYREF
  __int64 v31; // [rsp+78h] [rbp-78h]
  _BYTE v32[112]; // [rsp+80h] [rbp-70h] BYREF

  v2 = (__int64)(a1 + 25);
  v7 = (__int64 *)&v30;
  v30 = *(__int64 **)(v2 + 96);
  v26[0] = 6;
  v26[1] = 0;
  v3 = sub_24E84F0(v2, (__int64 *)&v30);
  result = v3[2];
  v27 = result;
  LOBYTE(v7) = result != 0;
  LOBYTE(v8) = result != -4096;
  if ( ((unsigned __int8)v8 & (result != 0)) == 0 || result == -8192 )
  {
    if ( !*(_QWORD *)(result + 16) )
      return result;
  }
  else
  {
    v7 = (__int64 *)(*v3 & 0xFFFFFFFFFFFFFFF8LL);
    sub_BD6050(v26, (unsigned __int64)v7);
    result = v27;
    if ( !*(_QWORD *)(v27 + 16) )
      goto LABEL_4;
  }
  v9 = a1[35];
  v30 = (__int64 *)v32;
  v31 = 0x800000000LL;
  if ( *(_DWORD *)(a1[3] + 280LL) == 3 )
  {
    if ( (*(_BYTE *)(v9 + 2) & 1) == 0 )
    {
      v10 = *(_QWORD *)(v9 + 96);
      goto LABEL_12;
    }
    sub_B2C6D0(v9, (__int64)v7, (__int64)v3, v8);
    v22 = a1[35];
    v10 = *(_QWORD *)(v9 + 96);
    v9 = v22;
    v23 = *(_BYTE *)(v22 + 2) & 1;
  }
  else
  {
    if ( (*(_BYTE *)(v9 + 2) & 1) == 0 )
    {
      v10 = *(_QWORD *)(v9 + 96) + 40LL;
      goto LABEL_12;
    }
    sub_B2C6D0(v9, (__int64)v7, (__int64)v3, v8);
    v22 = a1[35];
    v10 = *(_QWORD *)(v9 + 96) + 40LL;
    v9 = v22;
    v23 = *(_BYTE *)(v22 + 2) & 1;
  }
  if ( v23 )
    sub_B2C6D0(v9, (__int64)v7, v22, v21);
LABEL_12:
  v11 = *(_QWORD *)(v9 + 96) + 40LL * *(_QWORD *)(v9 + 104);
  for ( i = (unsigned int)v31; v10 != v11; LODWORD(v31) = v31 + 1 )
  {
    if ( i + 1 > (unsigned __int64)HIDWORD(v31) )
    {
      sub_C8D5F0((__int64)&v30, v32, i + 1, 8u, v4, v5);
      i = (unsigned int)v31;
    }
    v30[i] = v10;
    v10 += 40;
    i = (unsigned int)(v31 + 1);
  }
  if ( *(_BYTE *)(*(_QWORD *)(v27 + 8) + 8LL) != 15 )
  {
    sub_BD84D0(v27, *v30);
    goto LABEL_31;
  }
  v13 = *(_QWORD *)(v27 + 16);
  if ( !v13 )
    goto LABEL_31;
  do
  {
    while ( 1 )
    {
      v14 = v13;
      v13 = *(_QWORD *)(v13 + 8);
      v15 = *(_QWORD *)(v14 + 24);
      if ( *(_BYTE *)v15 == 93 && *(_DWORD *)(v15 + 80) == 1 )
        break;
      if ( !v13 )
        goto LABEL_23;
    }
    sub_BD84D0(*(_QWORD *)(v14 + 24), v30[**(unsigned int **)(v15 + 72)]);
    sub_B43D60((_QWORD *)v15);
  }
  while ( v13 );
LABEL_23:
  if ( !*(_QWORD *)(v27 + 16) )
  {
LABEL_31:
    v20 = (unsigned __int64)v30;
    if ( v30 == (__int64 *)v32 )
      goto LABEL_29;
    goto LABEL_28;
  }
  v16 = sub_ACADE0(*(__int64 ***)(v27 + 8));
  v17 = (__int64)v30;
  v18 = v16;
  v24 = &v30[(unsigned int)v31];
  if ( v24 != v30 )
  {
    v19 = 0;
    do
    {
      v25 = v19++;
      v17 += 8;
      v29 = 257;
      v18 = sub_2466140(a1 + 5, v18, *(_BYTE **)(v17 - 8), &v25, 1, (__int64)v28);
    }
    while ( v24 != (__int64 *)v17 );
  }
  sub_BD84D0(v27, v18);
  v20 = (unsigned __int64)v30;
  if ( v30 == (__int64 *)v32 )
    goto LABEL_29;
LABEL_28:
  _libc_free(v20);
LABEL_29:
  result = v27;
LABEL_4:
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0(v26);
  return result;
}
