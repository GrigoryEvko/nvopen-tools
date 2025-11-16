// Function: sub_27E28D0
// Address: 0x27e28d0
//
void __fastcall sub_27E28D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rcx
  __int64 v5; // rbx
  __int64 v6; // r12
  _BYTE *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // esi
  unsigned __int64 v12; // rdx
  _BYTE *v13; // r10
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rbx
  __int64 v17; // r13
  unsigned __int64 v18; // r14
  char v19; // bl
  int *v20; // r12
  unsigned __int8 *v21; // rdx
  unsigned __int8 *v22; // rsi
  int v23; // edx
  __int64 v25; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 v26; // [rsp+28h] [rbp-1A8h]
  __m128i v27; // [rsp+30h] [rbp-1A0h] BYREF
  _BYTE v28[32]; // [rsp+40h] [rbp-190h] BYREF
  _BYTE *v29; // [rsp+60h] [rbp-170h] BYREF
  __int64 v30; // [rsp+68h] [rbp-168h]
  _BYTE v31[264]; // [rsp+70h] [rbp-160h] BYREF
  int v32; // [rsp+178h] [rbp-58h] BYREF
  unsigned __int64 v33; // [rsp+180h] [rbp-50h]
  int *v34; // [rsp+188h] [rbp-48h]
  int *v35; // [rsp+190h] [rbp-40h]
  __int64 v36; // [rsp+198h] [rbp-38h]

  v3 = a2;
  v29 = v31;
  v30 = 0x1000000000LL;
  v32 = 0;
  v33 = 0;
  v34 = &v32;
  v35 = &v32;
  v36 = 0;
  sub_B129C0(&v25, a2);
  v5 = v25;
  v6 = v26;
  if ( v26 == v25 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      v15 = v5;
      v16 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
      LODWORD(v15) = (v15 >> 2) & 1;
      v17 = (unsigned int)v15;
      if ( (_DWORD)v15 )
        break;
      v7 = (_BYTE *)v16[17];
      if ( *v7 > 0x1Cu )
        goto LABEL_4;
      if ( !v16 )
        goto LABEL_14;
LABEL_10:
      v5 = (__int64)(v16 + 18);
      if ( v5 == v6 )
        goto LABEL_15;
    }
    v7 = *(_BYTE **)(*v16 + 136LL);
    if ( *v7 <= 0x1Cu )
      goto LABEL_14;
LABEL_4:
    v8 = *(unsigned int *)(*(_QWORD *)a1 + 24LL);
    if ( (_DWORD)v8 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
      v10 = (unsigned int)(v8 - 1);
      v11 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v12 = v9 + ((unsigned __int64)v11 << 6);
      v13 = *(_BYTE **)(v12 + 24);
      if ( v7 == v13 )
      {
LABEL_6:
        if ( v12 != v9 + (v8 << 6) )
        {
          v14 = *(_QWORD *)(v12 + 56);
          v27.m128i_i64[0] = (__int64)v7;
          v27.m128i_i64[1] = v14;
          sub_27E2720((__int64)v28, (__int64)&v29, &v27, v4, v9, v10);
        }
      }
      else
      {
        v23 = 1;
        while ( v13 != (_BYTE *)-4096LL )
        {
          v4 = (unsigned int)(v23 + 1);
          v11 = v10 & (v23 + v11);
          v12 = v9 + ((unsigned __int64)v11 << 6);
          v13 = *(_BYTE **)(v12 + 24);
          if ( v13 == v7 )
            goto LABEL_6;
          v23 = v4;
        }
      }
    }
    if ( !v17 && v16 )
      goto LABEL_10;
LABEL_14:
    v5 = (unsigned __int64)(v16 + 1) | 4;
  }
  while ( v5 != v6 );
LABEL_15:
  v3 = a2;
LABEL_16:
  if ( v36 )
  {
    v18 = (unsigned __int64)v34;
    v20 = &v32;
    v19 = 0;
  }
  else
  {
    v18 = (unsigned __int64)v29;
    v19 = 1;
    v20 = (int *)&v29[16 * (unsigned int)v30];
  }
  while ( (int *)v18 != v20 )
  {
    while ( v19 )
    {
      v21 = *(unsigned __int8 **)(v18 + 8);
      v22 = *(unsigned __int8 **)v18;
      v18 += 16LL;
      sub_B13360(v3, v22, v21, 0);
      if ( (int *)v18 == v20 )
        goto LABEL_21;
    }
    sub_B13360(v3, *(unsigned __int8 **)(v18 + 32), *(unsigned __int8 **)(v18 + 40), 0);
    v18 = sub_220EF30(v18);
  }
LABEL_21:
  sub_27DBC40(v33);
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
}
