// Function: sub_AA8210
// Address: 0xaa8210
//
__int64 __fastcall sub_AA8210(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // rbx
  _BYTE *v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 *v16; // rsi
  _BYTE *v17; // rdx
  __int64 *v18; // r15
  __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  int v21; // edx
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rsi
  __int64 *v26; // r13
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 *v34; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+18h] [rbp-78h] BYREF
  __int64 v37; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int16 v38; // [rsp+28h] [rbp-68h]
  __int64 *v39; // [rsp+30h] [rbp-60h] BYREF
  __int64 v40; // [rsp+38h] [rbp-58h]
  _BYTE v41[80]; // [rsp+40h] [rbp-50h] BYREF

  v33 = *(_QWORD *)(a1 + 72);
  v7 = sub_AA48A0(a1);
  v8 = sub_22077B0(80);
  v9 = v8;
  if ( v8 )
    sub_AA4D50(v8, v7, a4, v33, a1);
  if ( !a2 )
    BUG();
  v10 = a2[3];
  v36 = v10;
  if ( v10 )
    sub_B96E90(&v36, v10, 1);
  sub_AA80F0(v9, (unsigned __int64 *)(v9 + 48), 0, a1, *(__int64 **)(a1 + 56), 1, a2, a3);
  v11 = *(_QWORD *)(a1 + 16);
  if ( v11 )
  {
    while ( 1 )
    {
      v12 = *(_BYTE **)(v11 + 24);
      if ( (unsigned __int8)(*v12 - 30) <= 0xAu )
        break;
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        goto LABEL_39;
    }
    v13 = 0;
    v39 = (__int64 *)v41;
    v40 = 0x400000000LL;
    v14 = v11;
    while ( 1 )
    {
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v14 + 24) - 30) <= 0xAu )
      {
        v14 = *(_QWORD *)(v14 + 8);
        ++v13;
        if ( !v14 )
          goto LABEL_12;
      }
    }
LABEL_12:
    v15 = v13 + 1;
    v16 = (__int64 *)v41;
    if ( v15 > 4 )
    {
      sub_C8D5F0(&v39, v41, v15, 8);
      v12 = *(_BYTE **)(v11 + 24);
      v16 = &v39[(unsigned int)v40];
    }
    v17 = v12;
LABEL_17:
    if ( v16 )
      *v16 = *((_QWORD *)v17 + 5);
    while ( 1 )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        break;
      v17 = *(_BYTE **)(v11 + 24);
      if ( (unsigned __int8)(*v17 - 30) <= 0xAu )
      {
        ++v16;
        goto LABEL_17;
      }
    }
    LODWORD(v40) = v15 + v40;
    v34 = &v39[(unsigned int)v40];
    if ( v34 != v39 )
    {
      v18 = v39;
      do
      {
        v19 = *v18;
        v20 = *(_QWORD *)(*v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v20 == *v18 + 48 )
        {
          v22 = 0;
        }
        else
        {
          if ( !v20 )
            BUG();
          v21 = *(unsigned __int8 *)(v20 - 24);
          v22 = v20 - 24;
          if ( (unsigned int)(v21 - 30) >= 0xB )
            v22 = 0;
        }
        ++v18;
        sub_B47210(v22, a1, v9);
        sub_AA5D60(a1, v19, v9);
      }
      while ( v34 != v18 );
    }
  }
  else
  {
LABEL_39:
    v39 = (__int64 *)v41;
    v40 = 0x400000000LL;
  }
  sub_B43C20(&v37, v9);
  v32 = v37;
  v35 = v38;
  v23 = sub_BD2C40(72, 1);
  v24 = v23;
  if ( v23 )
    sub_B4C8F0(v23, a1, 1, v32, v35);
  v25 = v36;
  v26 = (__int64 *)(v24 + 48);
  v37 = v36;
  if ( !v36 )
  {
    if ( v26 == &v37 )
      goto LABEL_33;
    v25 = *(_QWORD *)(v24 + 48);
    if ( !v25 )
      goto LABEL_33;
LABEL_42:
    sub_B91220(v24 + 48);
    goto LABEL_43;
  }
  sub_B96E90(&v37, v36, 1);
  if ( v26 == &v37 )
  {
    v25 = v37;
    if ( v37 )
      sub_B91220(&v37);
    goto LABEL_33;
  }
  if ( *(_QWORD *)(v24 + 48) )
    goto LABEL_42;
LABEL_43:
  v25 = v37;
  *(_QWORD *)(v24 + 48) = v37;
  if ( v25 )
    sub_B976B0(&v37, v25, v24 + 48, v27, v28, v29);
LABEL_33:
  if ( v39 != (__int64 *)v41 )
    _libc_free(v39, v25);
  if ( v36 )
    sub_B91220(&v36);
  return v9;
}
