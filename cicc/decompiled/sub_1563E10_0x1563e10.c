// Function: sub_1563E10
// Address: 0x1563e10
//
__int64 __fastcall sub_1563E10(__int64 *a1, __int64 *a2, int *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 *v10; // rdi
  int v11; // edx
  const void *v12; // r10
  signed __int64 v13; // r8
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r12
  unsigned int v17; // edx
  __int64 *v18; // rsi
  int *v19; // r15
  int v20; // eax
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // r12
  unsigned __int64 v25; // rdx
  int v26; // r15d
  __int64 *v27; // rax
  __int64 *v28; // rdx
  signed __int64 v29; // [rsp+8h] [rbp-E8h]
  const void *v30; // [rsp+10h] [rbp-E0h]
  int *v32; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v33; // [rsp+28h] [rbp-C8h]
  __int64 *v34; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+38h] [rbp-B8h]
  _BYTE dest[32]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v37; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v38; // [rsp+78h] [rbp-78h]

  v8 = sub_15601B0(a1);
  v9 = sub_15601A0(a1);
  v10 = (__int64 *)dest;
  v11 = 0;
  v12 = (const void *)v9;
  v13 = v8 - v9;
  v34 = (__int64 *)dest;
  v35 = 0x400000000LL;
  v14 = (v8 - v9) >> 3;
  if ( (unsigned __int64)(v8 - v9) > 0x20 )
  {
    v29 = v8 - v9;
    v30 = (const void *)v9;
    sub_16CD150(&v34, dest, v13 >> 3, 8);
    v11 = v35;
    v13 = v29;
    v12 = v30;
    v10 = &v34[(unsigned int)v35];
  }
  if ( (const void *)v8 != v12 )
  {
    memcpy(v10, v12, v13);
    v11 = v35;
  }
  LODWORD(v15) = v14 + v11;
  LODWORD(v35) = v14 + v11;
  v16 = a4;
  v17 = a3[a4 - 1] + 2;
  if ( a3[a4 - 1] == -2 )
    v17 = 0;
  if ( v17 < (unsigned int)v15 )
    goto LABEL_8;
  v25 = v17 + 1;
  v15 = (unsigned int)v15;
  v26 = v25;
  if ( v25 < (unsigned int)v15 )
  {
    LODWORD(v35) = v25;
    goto LABEL_8;
  }
  if ( v25 <= (unsigned int)v15 )
  {
LABEL_8:
    v18 = v34;
    goto LABEL_9;
  }
  if ( v25 > HIDWORD(v35) )
  {
    v33 = v25;
    sub_16CD150(&v34, dest, v25, 8);
    v15 = (unsigned int)v35;
    v25 = v33;
  }
  v18 = v34;
  v27 = &v34[v15];
  v28 = &v34[v25];
  if ( v27 != v28 )
  {
    do
    {
      if ( v27 )
        *v27 = 0;
      ++v27;
    }
    while ( v28 != v27 );
    v18 = v34;
  }
  LODWORD(v35) = v26;
LABEL_9:
  v32 = &a3[v16];
  if ( &a3[v16] != a3 )
  {
    v19 = a3;
    do
    {
      v20 = *v19++;
      v21 = (unsigned int)(v20 + 2);
      sub_1563030(&v37, v18[v21]);
      sub_1562E30(&v37, a5);
      v22 = sub_1560BF0(a2, &v37);
      v34[v21] = v22;
      sub_155CC10(v38);
      v18 = v34;
    }
    while ( v32 != v19 );
  }
  v23 = sub_155F990(a2, v18, (unsigned int)v35);
  if ( v34 != (__int64 *)dest )
    _libc_free((unsigned __int64)v34);
  return v23;
}
