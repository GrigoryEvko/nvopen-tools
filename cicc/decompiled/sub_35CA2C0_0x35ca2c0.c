// Function: sub_35CA2C0
// Address: 0x35ca2c0
//
__int64 __fastcall sub_35CA2C0(__int64 *a1, __int64 a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r15
  __int64 v11; // r14
  char *v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 v16; // r13
  __int64 *v17; // r15
  __int64 v18; // r12
  char *v19; // rax
  int v20; // edx
  __int64 v21; // rsi
  int v22; // edx
  unsigned int v23; // edi
  __int64 *v24; // rax
  __int64 v25; // r11
  _QWORD *v26; // rax
  unsigned int v27; // r12d
  int v28; // eax
  __int64 *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-170h]
  __int64 v32; // [rsp+8h] [rbp-168h]
  unsigned int v33; // [rsp+1Ch] [rbp-154h]
  __int64 v34; // [rsp+20h] [rbp-150h] BYREF
  char *v35; // [rsp+28h] [rbp-148h]
  __int64 v36; // [rsp+30h] [rbp-140h]
  int v37; // [rsp+38h] [rbp-138h]
  unsigned __int8 v38; // [rsp+3Ch] [rbp-134h]
  char v39; // [rsp+40h] [rbp-130h] BYREF

  v35 = &v39;
  v7 = *a1;
  v8 = *((unsigned int *)a1 + 2);
  v34 = 0;
  v36 = 32;
  v38 = 1;
  v37 = 0;
  v31 = v7;
  v32 = v7 + 8 * v8;
  if ( v7 == v32 )
    return 0;
  v9 = 1;
  v10 = a2;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v32 - 8);
    if ( !(_BYTE)v9 )
    {
LABEL_35:
      sub_C8CC70((__int64)&v34, *(_QWORD *)(v32 - 8), (__int64)a3, v9, a5, a6);
      v9 = v38;
      goto LABEL_8;
    }
    v12 = v35;
    a3 = &v35[8 * HIDWORD(v36)];
    if ( v35 == a3 )
    {
LABEL_34:
      if ( HIDWORD(v36) >= (unsigned int)v36 )
        goto LABEL_35;
      ++HIDWORD(v36);
      *(_QWORD *)a3 = v11;
      v9 = v38;
      ++v34;
    }
    else
    {
      while ( v11 != *(_QWORD *)v12 )
      {
        v12 += 8;
        if ( a3 == v12 )
          goto LABEL_34;
      }
    }
LABEL_8:
    v13 = *(_QWORD *)(v11 + 112);
    v14 = *(unsigned int *)(v11 + 120);
    a6 = v13 + 8 * v14;
    if ( v13 != a6 )
      break;
LABEL_24:
    v32 -= 8;
    if ( v31 == v32 )
    {
      v27 = 0;
      goto LABEL_29;
    }
  }
  v15 = (__int64 *)(v13 + 8 * v14);
  v33 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
  v16 = v10;
  v17 = *(__int64 **)(v11 + 112);
  while ( 1 )
  {
    v18 = *v17;
    if ( (_BYTE)v9 )
    {
      a3 = &v35[8 * HIDWORD(v36)];
      while ( 1 )
      {
        v19 = v35;
        if ( v35 != a3 )
          break;
        if ( v15 == ++v17 )
          goto LABEL_23;
        v18 = *v17;
      }
      while ( v18 != *(_QWORD *)v19 )
      {
        v19 += 8;
        if ( a3 == v19 )
          goto LABEL_22;
      }
    }
    else
    {
      v30 = sub_C8CA60((__int64)&v34, *v17);
      v9 = v38;
      if ( !v30 )
        goto LABEL_22;
    }
    v20 = *(_DWORD *)(v16 + 24);
    v21 = *(_QWORD *)(v16 + 8);
    if ( !v20 )
      goto LABEL_28;
    v22 = v20 - 1;
    v23 = v22 & v33;
    v24 = (__int64 *)(v21 + 16LL * (v22 & v33));
    v25 = *v24;
    if ( v11 != *v24 )
      break;
LABEL_18:
    v26 = (_QWORD *)v24[1];
    if ( !v26 )
      goto LABEL_28;
    while ( 1 )
    {
      a3 = (char *)v26[4];
      if ( v18 == *(_QWORD *)a3 )
        break;
      v26 = (_QWORD *)*v26;
      if ( !v26 )
        goto LABEL_28;
    }
LABEL_22:
    if ( v15 == ++v17 )
    {
LABEL_23:
      v10 = v16;
      goto LABEL_24;
    }
  }
  v28 = 1;
  while ( v25 != -4096 )
  {
    a5 = (unsigned int)(v28 + 1);
    v23 = v22 & (v28 + v23);
    v24 = (__int64 *)(v21 + 16LL * v23);
    v25 = *v24;
    if ( v11 == *v24 )
      goto LABEL_18;
    v28 = a5;
  }
LABEL_28:
  v27 = 1;
LABEL_29:
  if ( !(_BYTE)v9 )
    _libc_free((unsigned __int64)v35);
  return v27;
}
