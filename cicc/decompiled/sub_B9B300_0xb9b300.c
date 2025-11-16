// Function: sub_B9B300
// Address: 0xb9b300
//
__int64 *__fastcall sub_B9B300(__int64 *a1, __int64 a2)
{
  __int64 v2; // r8
  unsigned __int8 v3; // al
  __int64 v4; // rbx
  __int64 v5; // rax
  char *v6; // r15
  char *v7; // rbx
  __int64 v8; // r10
  _QWORD *v9; // rax
  int v10; // edx
  _BYTE *v11; // r14
  unsigned __int8 v12; // al
  __int64 v13; // rbx
  const void *v14; // r15
  size_t v15; // rdx
  unsigned __int16 v16; // ax
  unsigned int v17; // r10d
  __int64 *v18; // rdi
  __int64 v19; // r11
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  _BYTE *v23; // rdi
  unsigned int v25; // [rsp+8h] [rbp-78h]
  size_t v26; // [rsp+10h] [rbp-70h]
  _BYTE *v28; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29; // [rsp+28h] [rbp-58h]
  _BYTE v30[80]; // [rsp+30h] [rbp-50h] BYREF

  v2 = a2 - 16;
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
  {
    v4 = *(_QWORD *)(a2 - 32);
    v5 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    v4 = v2 - 8LL * ((v3 >> 2) & 0xF);
    v5 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  }
  v6 = (char *)(v4 + 8 * v5);
  v7 = (char *)(v4 + 8);
  v29 = 0x400000000LL;
  v28 = v30;
  v8 = (v6 - v7) >> 3;
  if ( (unsigned __int64)(v6 - v7) > 0x20 )
  {
    sub_C8D5F0(&v28, v30, (v6 - v7) >> 3, 8);
    v11 = v28;
    v10 = v29;
    LODWORD(v8) = (v6 - v7) >> 3;
    v2 = a2 - 16;
    v9 = &v28[8 * (unsigned int)v29];
  }
  else
  {
    v9 = v30;
    v10 = 0;
    v11 = v30;
  }
  if ( v7 != v6 )
  {
    do
    {
      if ( v9 )
        *v9 = *(_QWORD *)v7;
      v7 += 8;
      ++v9;
    }
    while ( v7 != v6 );
    v11 = v28;
    v10 = v29;
  }
  v12 = *(_BYTE *)(a2 - 16);
  LODWORD(v29) = v10 + v8;
  v13 = (unsigned int)(v10 + v8);
  if ( (v12 & 2) != 0 )
  {
    v14 = **(const void ***)(a2 - 32);
    if ( v14 )
    {
LABEL_12:
      v14 = (const void *)sub_B91420((__int64)v14);
      goto LABEL_13;
    }
  }
  else
  {
    v14 = *(const void **)(v2 - 8LL * ((v12 >> 2) & 0xF));
    if ( v14 )
      goto LABEL_12;
  }
  v15 = 0;
LABEL_13:
  v26 = v15;
  v16 = sub_AF2710(a2);
  v17 = v16;
  v18 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v18 = (__int64 *)*v18;
  v19 = 0;
  if ( v26 )
  {
    v25 = v16;
    v20 = sub_B9B140(v18, v14, v26);
    v17 = v25;
    v19 = v20;
  }
  v21 = v17;
  v22 = sub_B029A0(v18, v17, v19, (__int64)v11, v13, 2u, 1);
  v23 = v28;
  *a1 = v22;
  if ( v23 != v30 )
    _libc_free(v23, v21);
  return a1;
}
