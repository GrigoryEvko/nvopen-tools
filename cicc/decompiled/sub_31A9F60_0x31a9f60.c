// Function: sub_31A9F60
// Address: 0x31a9f60
//
__int64 __fastcall sub_31A9F60(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // r13
  char v9; // di
  __int64 v10; // r14
  __int64 v11; // rsi
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r14
  __int64 v16; // rbx
  __int64 *v17; // r15
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 i; // r14
  __int64 j; // rbx
  __int64 v22; // rsi
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  __int64 *v25; // rbx
  __int64 *v26; // r14
  __int64 *v27; // rcx
  unsigned int v28; // r13d
  __int64 *v30; // rax
  __int64 *v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  __int64 v35; // rsi
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  __int64 *v38; // [rsp+8h] [rbp-158h]
  __int64 v39; // [rsp+10h] [rbp-150h] BYREF
  __int64 *v40; // [rsp+18h] [rbp-148h]
  __int64 v41; // [rsp+20h] [rbp-140h]
  int v42; // [rsp+28h] [rbp-138h]
  char v43; // [rsp+2Ch] [rbp-134h]
  char v44; // [rsp+30h] [rbp-130h] BYREF
  __int64 v45; // [rsp+70h] [rbp-F0h] BYREF
  char *v46; // [rsp+78h] [rbp-E8h]
  __int64 v47; // [rsp+80h] [rbp-E0h]
  int v48; // [rsp+88h] [rbp-D8h]
  char v49; // [rsp+8Ch] [rbp-D4h]
  char v50; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v51; // [rsp+D0h] [rbp-90h] BYREF
  char *v52; // [rsp+D8h] [rbp-88h]
  __int64 v53; // [rsp+E0h] [rbp-80h]
  int v54; // [rsp+E8h] [rbp-78h]
  char v55; // [rsp+ECh] [rbp-74h]
  char v56; // [rsp+F0h] [rbp-70h] BYREF

  v7 = *(unsigned int *)(a1 + 120);
  v8 = *(_QWORD *)(a1 + 112);
  v39 = 0;
  v40 = (__int64 *)&v44;
  v9 = 1;
  v43 = 1;
  v41 = 8;
  v42 = 0;
  v10 = v8 + 184 * v7;
  if ( v8 != v10 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(v8 + 40);
        if ( v9 )
          break;
LABEL_30:
        v8 += 184;
        sub_C8CC70((__int64)&v39, v11, (__int64)a3, a4, a5, a6);
        v9 = v43;
        if ( v8 == v10 )
          goto LABEL_8;
      }
      v12 = v40;
      a4 = HIDWORD(v41);
      a3 = &v40[HIDWORD(v41)];
      if ( v40 == a3 )
      {
LABEL_32:
        if ( HIDWORD(v41) >= (unsigned int)v41 )
          goto LABEL_30;
        a4 = (unsigned int)(HIDWORD(v41) + 1);
        v8 += 184;
        ++HIDWORD(v41);
        *a3 = v11;
        v9 = v43;
        ++v39;
        if ( v8 == v10 )
          break;
      }
      else
      {
        while ( v11 != *v12 )
        {
          if ( a3 == ++v12 )
            goto LABEL_32;
        }
        v8 += 184;
        if ( v8 == v10 )
          break;
      }
    }
  }
LABEL_8:
  v13 = *(__int64 **)(a1 + 352);
  if ( *(_BYTE *)(a1 + 372) )
    v14 = *(unsigned int *)(a1 + 364);
  else
    v14 = *(unsigned int *)(a1 + 360);
  v15 = &v13[v14];
  if ( v13 != v15 )
  {
    while ( 1 )
    {
      v16 = *v13;
      v17 = v13;
      if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v15 == ++v13 )
        goto LABEL_13;
    }
    if ( v15 != v13 )
    {
      if ( !v43 )
        goto LABEL_52;
LABEL_42:
      v30 = v40;
      v31 = &v40[HIDWORD(v41)];
      if ( v40 != v31 )
      {
        do
        {
          if ( *v30 == v16 )
            goto LABEL_46;
          ++v30;
        }
        while ( v31 != v30 );
      }
LABEL_53:
      v33 = *(_QWORD *)(v16 + 16);
      if ( !v33 )
        goto LABEL_46;
      do
      {
        v34 = *(_QWORD *)a1;
        v35 = *(_QWORD *)(*(_QWORD *)(v33 + 24) + 40LL);
        if ( *(_BYTE *)(*(_QWORD *)a1 + 84LL) )
        {
          v36 = *(_QWORD **)(v34 + 64);
          v37 = &v36[*(unsigned int *)(v34 + 76)];
          if ( v36 == v37 )
          {
LABEL_36:
            v28 = 0;
            goto LABEL_37;
          }
          while ( v35 != *v36 )
          {
            if ( v37 == ++v36 )
              goto LABEL_36;
          }
        }
        else if ( !sub_C8CA60(v34 + 56, v35) )
        {
          goto LABEL_36;
        }
        v33 = *(_QWORD *)(v33 + 8);
      }
      while ( v33 );
LABEL_46:
      while ( 1 )
      {
        v32 = v17 + 1;
        if ( v17 + 1 == v15 )
          break;
        v16 = *v32;
        for ( ++v17; (unsigned __int64)*v32 >= 0xFFFFFFFFFFFFFFFELL; v17 = v32 )
        {
          if ( v15 == ++v32 )
            goto LABEL_13;
          v16 = *v32;
        }
        if ( v15 == v17 )
          break;
        if ( v43 )
          goto LABEL_42;
LABEL_52:
        if ( !sub_C8CA60((__int64)&v39, v16) )
          goto LABEL_53;
      }
    }
  }
LABEL_13:
  v18 = *(_QWORD *)(a1 + 160);
  v19 = *(_QWORD *)a1;
  for ( i = v18 + 88LL * *(unsigned int *)(a1 + 168); i != v18; v18 += 88 )
  {
    for ( j = *(_QWORD *)(*(_QWORD *)v18 + 16LL); j; v19 = *(_QWORD *)a1 )
    {
      while ( 1 )
      {
        v22 = *(_QWORD *)(*(_QWORD *)(j + 24) + 40LL);
        if ( !*(_BYTE *)(v19 + 84) )
          break;
        v23 = *(_QWORD **)(v19 + 64);
        v24 = &v23[*(unsigned int *)(v19 + 76)];
        if ( v23 == v24 )
          goto LABEL_36;
        while ( v22 != *v23 )
        {
          if ( v24 == ++v23 )
            goto LABEL_36;
        }
        j = *(_QWORD *)(j + 8);
        if ( !j )
          goto LABEL_21;
      }
      if ( !sub_C8CA60(v19 + 56, v22) )
        goto LABEL_36;
      j = *(_QWORD *)(j + 8);
    }
LABEL_21:
    ;
  }
  v25 = *(__int64 **)(v19 + 32);
  v26 = *(__int64 **)(v19 + 40);
  v45 = 0;
  v46 = &v50;
  v47 = 8;
  v48 = 0;
  v49 = 1;
  v51 = 0;
  v52 = &v56;
  v53 = 8;
  v54 = 0;
  v55 = 1;
  if ( v25 == v26 )
  {
    v28 = 1;
  }
  else
  {
    v27 = &v51;
    do
    {
      v38 = v27;
      v28 = sub_31A9620(a1, *v25, &v45, (__int64)v27, a5, a6);
      if ( !(_BYTE)v28 )
        break;
      ++v25;
      v27 = v38;
    }
    while ( v26 != v25 );
    if ( !v55 )
      _libc_free((unsigned __int64)v52);
  }
  if ( !v49 )
    _libc_free((unsigned __int64)v46);
LABEL_37:
  if ( !v43 )
    _libc_free((unsigned __int64)v40);
  return v28;
}
