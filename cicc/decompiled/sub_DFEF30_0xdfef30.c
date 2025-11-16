// Function: sub_DFEF30
// Address: 0xdfef30
//
__int64 __fastcall sub_DFEF30(__int64 a1, __int64 a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  char v9; // r12
  __int64 v10; // rbx
  char *v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  unsigned int v14; // eax
  unsigned int v15; // r13d
  unsigned int v16; // edx
  unsigned int v17; // eax
  int v18; // ebx
  __int64 v19; // rax
  char *v20; // rax
  int v21; // ecx
  int v22; // ecx
  unsigned int v23; // edi
  __int64 *v24; // rax
  __int64 v25; // r10
  _QWORD *v26; // rax
  unsigned int v27; // r13d
  int v28; // eax
  __int64 *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-180h]
  __int64 v32; // [rsp+8h] [rbp-178h]
  int v33; // [rsp+1Ch] [rbp-164h]
  __int64 v34; // [rsp+20h] [rbp-160h]
  char *v35; // [rsp+28h] [rbp-158h]
  __int64 v36; // [rsp+30h] [rbp-150h] BYREF
  char *v37; // [rsp+38h] [rbp-148h]
  __int64 v38; // [rsp+40h] [rbp-140h]
  int v39; // [rsp+48h] [rbp-138h]
  char v40; // [rsp+4Ch] [rbp-134h]
  char v41; // [rsp+50h] [rbp-130h] BYREF

  v37 = &v41;
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_QWORD *)(a1 + 48);
  v36 = 0;
  v38 = 32;
  v39 = 0;
  v40 = 1;
  v32 = v8;
  v31 = v7;
  if ( v8 == v7 )
    return 0;
  v9 = 1;
  while ( 1 )
  {
    v10 = *(_QWORD *)(v32 - 8);
    if ( !v9 )
      goto LABEL_36;
    v11 = v37;
    a4 = HIDWORD(v38);
    a3 = &v37[8 * HIDWORD(v38)];
    if ( v37 == a3 )
    {
LABEL_35:
      if ( HIDWORD(v38) < (unsigned int)v38 )
      {
        a4 = (unsigned int)++HIDWORD(v38);
        *(_QWORD *)a3 = v10;
        v9 = v40;
        ++v36;
        goto LABEL_8;
      }
LABEL_36:
      v7 = *(_QWORD *)(v32 - 8);
      sub_C8CC70((__int64)&v36, v7, (__int64)a3, a4, a5, a6);
      v9 = v40;
      goto LABEL_8;
    }
    while ( v10 != *(_QWORD *)v11 )
    {
      v11 += 8;
      if ( a3 == v11 )
        goto LABEL_35;
    }
LABEL_8:
    a3 = (char *)(v10 + 48);
    v12 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v12 != v10 + 48 )
    {
      if ( !v12 )
        BUG();
      v13 = v12 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 <= 0xA )
      {
        v14 = sub_B46E30(v13);
        v7 = v14;
        if ( v14 )
          break;
      }
    }
LABEL_25:
    v32 -= 8;
    if ( v31 == v32 )
    {
      v27 = 0;
      goto LABEL_30;
    }
  }
  v34 = v10;
  v15 = 0;
  v16 = (unsigned int)v10 >> 4;
  v17 = (unsigned int)v10 >> 9;
  v18 = v7;
  v33 = v17 ^ v16;
  while ( 1 )
  {
    v7 = v15;
    v19 = sub_B46EC0(v13, v15);
    a3 = (char *)v19;
    if ( v9 )
    {
      v20 = v37;
      a4 = (__int64)&v37[8 * HIDWORD(v38)];
      if ( v37 == (char *)a4 )
        goto LABEL_24;
      while ( a3 != *(char **)v20 )
      {
        v20 += 8;
        if ( (char *)a4 == v20 )
          goto LABEL_24;
      }
    }
    else
    {
      v7 = v19;
      v35 = (char *)v19;
      v30 = sub_C8CA60((__int64)&v36, v19);
      v9 = v40;
      a3 = v35;
      if ( !v30 )
        goto LABEL_24;
    }
    v21 = *(_DWORD *)(a2 + 24);
    v7 = *(_QWORD *)(a2 + 8);
    if ( !v21 )
      goto LABEL_29;
    v22 = v21 - 1;
    v23 = v22 & v33;
    v24 = (__int64 *)(v7 + 16LL * (v22 & (unsigned int)v33));
    v25 = *v24;
    if ( v34 != *v24 )
      break;
LABEL_20:
    v26 = (_QWORD *)v24[1];
    if ( !v26 )
      goto LABEL_29;
    while ( 1 )
    {
      a4 = v26[4];
      if ( a3 == *(char **)a4 )
        break;
      v26 = (_QWORD *)*v26;
      if ( !v26 )
        goto LABEL_29;
    }
LABEL_24:
    if ( v18 == ++v15 )
      goto LABEL_25;
  }
  v28 = 1;
  while ( v25 != -4096 )
  {
    a5 = (unsigned int)(v28 + 1);
    v23 = v22 & (v28 + v23);
    v24 = (__int64 *)(v7 + 16LL * v23);
    v25 = *v24;
    if ( v34 == *v24 )
      goto LABEL_20;
    v28 = a5;
  }
LABEL_29:
  v27 = 1;
LABEL_30:
  if ( !v9 )
    _libc_free(v37, v7);
  return v27;
}
