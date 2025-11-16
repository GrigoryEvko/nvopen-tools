// Function: sub_2D06650
// Address: 0x2d06650
//
__int64 __fastcall sub_2D06650(__int64 *a1, __int64 a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  char v9; // r12
  __int64 v10; // rbx
  char *v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  int v14; // esi
  unsigned int v15; // r13d
  unsigned int v16; // edx
  unsigned int v17; // eax
  int v18; // ebx
  __int64 v19; // rax
  char *v20; // rax
  int v21; // ecx
  __int64 v22; // rsi
  int v23; // ecx
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // r10
  _QWORD *v27; // rax
  unsigned int v28; // r13d
  int v29; // eax
  __int64 *v31; // rax
  __int64 v32; // [rsp+0h] [rbp-180h]
  __int64 v33; // [rsp+8h] [rbp-178h]
  int v34; // [rsp+1Ch] [rbp-164h]
  __int64 v35; // [rsp+20h] [rbp-160h]
  char *v36; // [rsp+28h] [rbp-158h]
  __int64 v37; // [rsp+30h] [rbp-150h] BYREF
  char *v38; // [rsp+38h] [rbp-148h]
  __int64 v39; // [rsp+40h] [rbp-140h]
  int v40; // [rsp+48h] [rbp-138h]
  char v41; // [rsp+4Ch] [rbp-134h]
  char v42; // [rsp+50h] [rbp-130h] BYREF

  v38 = &v42;
  v7 = *a1;
  v8 = *((unsigned int *)a1 + 2);
  v37 = 0;
  v39 = 32;
  v41 = 1;
  v40 = 0;
  v32 = v7;
  v33 = v7 + 8 * v8;
  if ( v7 == v33 )
    return 0;
  v9 = 1;
  while ( 1 )
  {
    v10 = *(_QWORD *)(v33 - 8);
    if ( !v9 )
      goto LABEL_36;
    v11 = v38;
    a4 = HIDWORD(v39);
    a3 = &v38[8 * HIDWORD(v39)];
    if ( v38 == a3 )
    {
LABEL_35:
      if ( HIDWORD(v39) < (unsigned int)v39 )
      {
        a4 = (unsigned int)++HIDWORD(v39);
        *(_QWORD *)a3 = v10;
        v9 = v41;
        ++v37;
        goto LABEL_8;
      }
LABEL_36:
      sub_C8CC70((__int64)&v37, *(_QWORD *)(v33 - 8), (__int64)a3, a4, a5, a6);
      v9 = v41;
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
        if ( v14 )
          break;
      }
    }
LABEL_25:
    v33 -= 8;
    if ( v32 == v33 )
    {
      v28 = 0;
      goto LABEL_30;
    }
  }
  v35 = v10;
  v15 = 0;
  v16 = (unsigned int)v10 >> 4;
  v17 = (unsigned int)v10 >> 9;
  v18 = v14;
  v34 = v17 ^ v16;
  while ( 1 )
  {
    v19 = sub_B46EC0(v13, v15);
    a3 = (char *)v19;
    if ( v9 )
    {
      v20 = v38;
      a4 = (__int64)&v38[8 * HIDWORD(v39)];
      if ( v38 == (char *)a4 )
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
      v36 = (char *)v19;
      v31 = sub_C8CA60((__int64)&v37, v19);
      v9 = v41;
      a3 = v36;
      if ( !v31 )
        goto LABEL_24;
    }
    v21 = *(_DWORD *)(a2 + 24);
    v22 = *(_QWORD *)(a2 + 8);
    if ( !v21 )
      goto LABEL_29;
    v23 = v21 - 1;
    v24 = v23 & v34;
    v25 = (__int64 *)(v22 + 16LL * (v23 & (unsigned int)v34));
    v26 = *v25;
    if ( v35 != *v25 )
      break;
LABEL_20:
    v27 = (_QWORD *)v25[1];
    if ( !v27 )
      goto LABEL_29;
    while ( 1 )
    {
      a4 = v27[4];
      if ( a3 == *(char **)a4 )
        break;
      v27 = (_QWORD *)*v27;
      if ( !v27 )
        goto LABEL_29;
    }
LABEL_24:
    if ( v18 == ++v15 )
      goto LABEL_25;
  }
  v29 = 1;
  while ( v26 != -4096 )
  {
    a5 = (unsigned int)(v29 + 1);
    v24 = v23 & (v29 + v24);
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v35 == *v25 )
      goto LABEL_20;
    v29 = a5;
  }
LABEL_29:
  v28 = 1;
LABEL_30:
  if ( !v9 )
    _libc_free((unsigned __int64)v38);
  return v28;
}
