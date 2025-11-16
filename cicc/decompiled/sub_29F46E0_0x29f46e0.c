// Function: sub_29F46E0
// Address: 0x29f46e0
//
void __fastcall sub_29F46E0(__int64 a1, __int64 a2, char *a3, __int64 a4, char *a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // rax
  char **v14; // r15
  __int64 v15; // r9
  __int64 v16; // rdx
  _BYTE *v17; // r10
  _BYTE *v18; // rax
  bool v19; // cc
  _BYTE *v20; // r11
  __int64 v21; // rcx
  _BYTE *v22; // rdx
  char **v23; // r14
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // rdx
  char *v27; // rax
  __int64 v28; // r10
  __int64 v29; // rdi
  __int64 v30; // r8
  __int64 v31; // r13
  __int64 v32; // rax
  char *v33; // [rsp+8h] [rbp-E8h]
  __int64 v35; // [rsp+18h] [rbp-D8h]
  char *v36; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v37; // [rsp+28h] [rbp-C8h]
  __int64 v38; // [rsp+30h] [rbp-C0h]
  _BYTE v39[184]; // [rsp+38h] [rbp-B8h] BYREF

  v8 = a1 + 152 * a2;
  v9 = (__int64)(a3 - 1) / 2;
  v33 = a3;
  v35 = v9;
  if ( a2 < v9 )
  {
    v10 = a2;
    while ( 1 )
    {
      v11 = v10 + 1;
      v10 = 2 * (v10 + 1);
      v12 = 8 * (v10 + 2 * (v10 + 16 * v11));
      v13 = (_QWORD *)(a1 + v12 - 152);
      v14 = (char **)(a1 + v12);
      v15 = v13[1];
      v16 = *(_QWORD *)(a1 + v12 + 8);
      v17 = (_BYTE *)*v13;
      v18 = *(_BYTE **)(a1 + v12);
      v19 = v15 < v16;
      v20 = &v18[v16];
      v21 = (__int64)&v18[v15];
      v22 = v17;
      if ( !v19 )
        v21 = (__int64)v20;
      if ( v18 != (_BYTE *)v21 )
        break;
LABEL_13:
      v15 += (__int64)v17;
      if ( v22 != (_BYTE *)v15 )
        goto LABEL_10;
LABEL_11:
      v23 = v14;
      sub_29F3DD0(v8, v14, (__int64)v22, v21, (__int64)a5, v15);
      if ( v10 >= v35 )
        goto LABEL_16;
      v8 = (__int64)v14;
    }
    while ( *v18 >= *v22 )
    {
      if ( *v18 > *v22 )
        goto LABEL_11;
      ++v18;
      ++v22;
      if ( (_BYTE *)v21 == v18 )
        goto LABEL_13;
    }
LABEL_10:
    --v10;
    v14 = (char **)(a1 + 152 * v10);
    goto LABEL_11;
  }
  v23 = (char **)v8;
  v10 = a2;
LABEL_16:
  if ( ((unsigned __int8)v33 & 1) == 0 )
  {
    a3 = v33 - 2;
    if ( (__int64)(v33 - 2) / 2 == v10 )
    {
      v30 = v10 + 1;
      v31 = 2 * (v10 + 1);
      v30 *= 16;
      v32 = v31 + 2 * (v30 + v31);
      v10 = v31 - 1;
      sub_29F3DD0((__int64)v23, (char **)(a1 + 8 * v32 - 152), (__int64)a3, v9, v30, a6);
      v23 = (char **)(a1 + 152 * v10);
    }
  }
  v37 = 0;
  v36 = v39;
  v38 = 128;
  if ( *(_QWORD *)(a4 + 8) )
    sub_29F3DD0((__int64)&v36, (char **)a4, (__int64)a3, v9, (__int64)a5, a6);
  v24 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      a5 = v36;
      v25 = a1 + 152 * v24;
      v26 = *(_QWORD *)(v25 + 8);
      v27 = *(char **)v25;
      v19 = v37 < v26;
      v28 = *(_QWORD *)v25 + v26;
      v9 = *(_QWORD *)v25 + v37;
      a3 = v36;
      if ( !v19 )
        v9 = v28;
      if ( v27 == (char *)v9 )
      {
LABEL_31:
        if ( a3 == &v36[v37] )
          break;
      }
      else
      {
        while ( *v27 >= *a3 )
        {
          if ( *v27 > *a3 )
            goto LABEL_32;
          ++v27;
          ++a3;
          if ( (char *)v9 == v27 )
            goto LABEL_31;
        }
      }
      v29 = (__int64)v23;
      v23 = (char **)(a1 + 152 * v24);
      sub_29F3DD0(v29, v23, (__int64)a3, v9, (__int64)v36, a6);
      a3 = (char *)(v24 - 1);
      if ( a2 >= v24 )
        break;
      v24 = (v24 - 1) / 2;
    }
  }
LABEL_32:
  sub_29F3DD0((__int64)v23, &v36, (__int64)a3, v9, (__int64)a5, a6);
  if ( v36 != v39 )
    _libc_free((unsigned __int64)v36);
}
