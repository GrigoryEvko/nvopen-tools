// Function: sub_B57640
// Address: 0xb57640
//
__int64 __fastcall sub_B57640(__int64 a1, __int64 *a2, __int64 a3, unsigned __int16 a4)
{
  bool v5; // sf
  int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r14
  char v12; // r13
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // r13
  __int64 v23; // rbx
  __int64 *v24; // r12
  __int64 v25; // rdi
  __m128i *v27; // rax
  __int64 v28; // rdi
  int v29; // edx
  __int64 v30; // rax
  int v31; // [rsp+8h] [rbp-D8h]
  __m128i *v32; // [rsp+10h] [rbp-D0h]
  __m128i *v33; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+38h] [rbp-A8h] BYREF
  _QWORD v37[4]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 *v38; // [rsp+60h] [rbp-80h] BYREF
  __int64 v39; // [rsp+68h] [rbp-78h]
  _BYTE v40[112]; // [rsp+70h] [rbp-70h] BYREF

  v38 = (__int64 *)v40;
  v5 = *(char *)(a1 + 7) < 0;
  v39 = 0x100000000LL;
  if ( !v5 )
    goto LABEL_16;
  v6 = (int)a2;
  v7 = sub_BD2BC0(a1);
  v9 = v7 + v8;
  if ( *(char *)(a1 + 7) >= 0 )
    v10 = v9 >> 4;
  else
    LODWORD(v10) = (v9 - sub_BD2BC0(a1)) >> 4;
  v11 = 0;
  v12 = 0;
  v13 = 16LL * (unsigned int)v10;
  if ( !(_DWORD)v10 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      v16 = 0;
      if ( *(char *)(a1 + 7) < 0 )
        v16 = sub_BD2BC0(a1);
      v17 = (__int64 *)(v11 + v16);
      v18 = *((unsigned int *)v17 + 2);
      v19 = *v17;
      v20 = *((unsigned int *)v17 + 3);
      a2 = (__int64 *)(32 * v18);
      v21 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v37[2] = v19;
      v37[0] = (char *)&a2[v21 / 0xFFFFFFFFFFFFFFF8LL] + a1;
      v37[1] = (32 * v20 - (__int64)a2) >> 5;
      if ( v6 != *(_DWORD *)(v19 + 8) )
        break;
      v11 += 16;
      v12 = 1;
      if ( v13 == v11 )
        goto LABEL_15;
    }
    v14 = v39;
    if ( HIDWORD(v39) <= (unsigned int)v39 )
    {
      v27 = (__m128i *)sub_C8D7D0(&v38, v40, 0, 56, &v36);
      v28 = (__int64)&v27->m128i_i64[7 * (unsigned int)v39];
      if ( v28 )
      {
        v32 = v27;
        sub_B56460(v28, (__int64)v37);
        v27 = v32;
      }
      a2 = (__int64 *)v27;
      v33 = v27;
      sub_B56820((__int64)&v38, v27);
      v29 = v36;
      v30 = (__int64)v33;
      if ( v38 != (__int64 *)v40 )
      {
        v31 = v36;
        _libc_free(v38, a2);
        v29 = v31;
        v30 = (__int64)v33;
      }
      LODWORD(v39) = v39 + 1;
      v38 = (__int64 *)v30;
      HIDWORD(v39) = v29;
    }
    else
    {
      v15 = (__int64)&v38[7 * (unsigned int)v39];
      if ( v15 )
      {
        a2 = v37;
        sub_B56460(v15, (__int64)v37);
        v14 = v39;
      }
      LODWORD(v39) = v14 + 1;
    }
    v11 += 16;
  }
  while ( v13 != v11 );
LABEL_15:
  if ( v12 )
  {
    a2 = v38;
    v22 = sub_B4BA60((unsigned __int8 *)a1, (__int64)v38, (unsigned int)v39, a3, a4);
  }
  else
  {
LABEL_16:
    v22 = a1;
  }
  v23 = (__int64)v38;
  v24 = &v38[7 * (unsigned int)v39];
  if ( v38 != v24 )
  {
    do
    {
      v25 = *(v24 - 3);
      v24 -= 7;
      if ( v25 )
      {
        a2 = (__int64 *)(v24[6] - v25);
        j_j___libc_free_0(v25, a2);
      }
      if ( (__int64 *)*v24 != v24 + 2 )
      {
        a2 = (__int64 *)(v24[2] + 1);
        j_j___libc_free_0(*v24, a2);
      }
    }
    while ( (__int64 *)v23 != v24 );
    v24 = v38;
  }
  if ( v24 != (__int64 *)v40 )
    _libc_free(v24, a2);
  return v22;
}
