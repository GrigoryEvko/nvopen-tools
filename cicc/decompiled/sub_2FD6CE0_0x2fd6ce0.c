// Function: sub_2FD6CE0
// Address: 0x2fd6ce0
//
__int64 __fastcall sub_2FD6CE0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r12
  char v8; // di
  __int64 *v9; // r14
  __int64 v10; // rsi
  __int64 *v11; // rax
  unsigned __int64 v12; // rbx
  const void *v13; // r14
  __int64 v14; // r12
  __int64 *v15; // rdi
  __int64 *v16; // r12
  __int64 *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 *v21; // r14
  __int64 *v22; // r13
  __int64 v23; // r15
  __int64 *v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rdi
  __int64 (*v27)(); // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-1E8h]
  __int64 v38; // [rsp+10h] [rbp-1E0h]
  unsigned __int8 v39; // [rsp+18h] [rbp-1D8h]
  __int64 v40; // [rsp+18h] [rbp-1D8h]
  __int64 v41; // [rsp+28h] [rbp-1C8h]
  __int64 v43; // [rsp+48h] [rbp-1A8h] BYREF
  __int64 v44; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v45; // [rsp+58h] [rbp-198h] BYREF
  __int64 *v46; // [rsp+60h] [rbp-190h] BYREF
  __int64 v47; // [rsp+68h] [rbp-188h]
  _BYTE v48[64]; // [rsp+70h] [rbp-180h] BYREF
  __int64 v49; // [rsp+B0h] [rbp-140h] BYREF
  __int64 *v50; // [rsp+B8h] [rbp-138h]
  __int64 v51; // [rsp+C0h] [rbp-130h]
  int v52; // [rsp+C8h] [rbp-128h]
  char v53; // [rsp+CCh] [rbp-124h]
  char v54; // [rsp+D0h] [rbp-120h] BYREF
  _BYTE *v55; // [rsp+110h] [rbp-E0h] BYREF
  __int64 v56; // [rsp+118h] [rbp-D8h]
  _BYTE v57[208]; // [rsp+120h] [rbp-D0h] BYREF

  v7 = *(__int64 **)(a2 + 112);
  v38 = (__int64)a3;
  v8 = 1;
  v9 = &v7[*(unsigned int *)(a2 + 120)];
  v49 = 0;
  v50 = (__int64 *)&v54;
  v51 = 8;
  v52 = 0;
  v53 = 1;
  if ( v7 != v9 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = *v7;
        if ( v8 )
          break;
LABEL_36:
        ++v7;
        sub_C8CC70((__int64)&v49, v10, (__int64)a3, a4, a5, a6);
        v8 = v53;
        if ( v9 == v7 )
          goto LABEL_8;
      }
      v11 = v50;
      a4 = HIDWORD(v51);
      a3 = &v50[HIDWORD(v51)];
      if ( v50 == a3 )
      {
LABEL_38:
        if ( HIDWORD(v51) >= (unsigned int)v51 )
          goto LABEL_36;
        a4 = (unsigned int)(HIDWORD(v51) + 1);
        ++v7;
        ++HIDWORD(v51);
        *a3 = v10;
        v8 = v53;
        ++v49;
        if ( v9 == v7 )
          break;
      }
      else
      {
        while ( v10 != *v11 )
        {
          if ( a3 == ++v11 )
            goto LABEL_38;
        }
        if ( v9 == ++v7 )
          break;
      }
    }
  }
LABEL_8:
  v12 = *(unsigned int *)(a2 + 72);
  v13 = *(const void **)(a2 + 64);
  v46 = (__int64 *)v48;
  v14 = 8 * v12;
  v47 = 0x800000000LL;
  if ( v12 > 8 )
  {
    sub_C8D5F0((__int64)&v46, v48, v12, 8u, a5, a6);
    v15 = &v46[(unsigned int)v47];
  }
  else
  {
    v15 = (__int64 *)v48;
    if ( !v14 )
      goto LABEL_10;
  }
  memcpy(v15, v13, 8 * v12);
  v15 = v46;
  LODWORD(v14) = v47;
LABEL_10:
  LODWORD(v47) = v14 + v12;
  v16 = &v15[(unsigned int)(v14 + v12)];
  if ( v16 != v15 )
  {
    v39 = 0;
    v17 = v15;
    v41 = a2;
    while ( 1 )
    {
      v18 = *v17;
      if ( (unsigned __int8)sub_2E31A70(*v17) || (unsigned __int8)sub_2E31AC0(v18) )
        goto LABEL_27;
      v19 = *(_QWORD *)(v18 + 112);
      v20 = *(unsigned int *)(v18 + 120);
      if ( v19 == v19 + 8 * v20 )
        goto LABEL_26;
      v37 = v18;
      v21 = *(__int64 **)(v18 + 112);
      v22 = (__int64 *)(v19 + 8 * v20);
      do
      {
        v23 = *v21;
        if ( v53 )
        {
          v24 = v50;
          v25 = &v50[HIDWORD(v51)];
          if ( v50 == v25 )
            goto LABEL_24;
          while ( v23 != *v24 )
          {
            if ( v25 == ++v24 )
              goto LABEL_24;
          }
        }
        else if ( !sub_C8CA60((__int64)&v49, v23) )
        {
          goto LABEL_24;
        }
        if ( v23 + 48 != (*(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL)
          && (*(_WORD *)(*(_QWORD *)(v23 + 56) + 68LL) == 68 || !*(_WORD *)(*(_QWORD *)(v23 + 56) + 68LL)) )
        {
          goto LABEL_27;
        }
LABEL_24:
        ++v21;
      }
      while ( v22 != v21 );
      v18 = v37;
LABEL_26:
      v43 = 0;
      v56 = 0x400000000LL;
      v44 = 0;
      v26 = *a1;
      v55 = v57;
      v27 = *(__int64 (**)())(*(_QWORD *)v26 + 344LL);
      if ( v27 == sub_2DB1AE0 )
      {
LABEL_27:
        if ( v16 == ++v17 )
          goto LABEL_28;
        continue;
      }
      if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v27)(
             v26,
             v18,
             &v43,
             &v44,
             &v55,
             0) )
      {
        if ( v55 == v57 )
          goto LABEL_27;
        _libc_free((unsigned __int64)v55);
        if ( v16 == ++v17 )
          goto LABEL_28;
        continue;
      }
      v29 = v43;
      v30 = **(_QWORD **)(v41 + 112);
      v31 = *(_QWORD *)(v18 + 8);
      if ( v31 == *(_QWORD *)(v18 + 32) + 320LL )
        v31 = 0;
      if ( !(_DWORD)v56 )
      {
        v44 = v43;
        v32 = v43;
        if ( v43 )
          goto LABEL_50;
LABEL_71:
        v43 = v31;
        v29 = v31;
        goto LABEL_48;
      }
      v32 = v44;
      if ( !v43 )
        goto LABEL_71;
LABEL_48:
      if ( !v32 )
      {
        v44 = v31;
        v32 = v31;
      }
LABEL_50:
      if ( v41 == v32 )
      {
        v44 = v30;
        v32 = v30;
        if ( v41 == v29 )
          goto LABEL_77;
      }
      else
      {
        if ( v41 != v29 )
          goto LABEL_52;
LABEL_77:
        v43 = v30;
        v29 = v30;
      }
LABEL_52:
      if ( v32 == v29 )
      {
        LODWORD(v56) = 0;
        v44 = 0;
        if ( v31 )
        {
          if ( v31 == v32 )
            goto LABEL_79;
        }
        else if ( !v32 )
        {
          goto LABEL_79;
        }
      }
      else if ( v32 == v31 )
      {
        v44 = 0;
      }
      else if ( v31 == v29 && !v32 )
      {
LABEL_79:
        v43 = 0;
      }
      v40 = v30;
      sub_2E32880(&v45, v18);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)*a1 + 360LL))(*a1, v18, 0);
      if ( sub_2E322C0(v18, v40) )
        sub_2E33650(v18, v41);
      else
        sub_2E33690(v18, v41, v40);
      if ( v43 )
        (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, _BYTE *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)*a1 + 368LL))(
          *a1,
          v18,
          v43,
          v44,
          v55,
          (unsigned int)v56,
          &v45,
          0);
      v35 = *(unsigned int *)(v38 + 8);
      if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v38 + 12) )
      {
        sub_C8D5F0(v38, (const void *)(v38 + 16), v35 + 1, 8u, v33, v34);
        v35 = *(unsigned int *)(v38 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v38 + 8 * v35) = v18;
      v36 = v45;
      ++*(_DWORD *)(v38 + 8);
      if ( v36 )
        sub_B91220((__int64)&v45, v36);
      if ( v55 != v57 )
        _libc_free((unsigned __int64)v55);
      ++v17;
      v39 = 1;
      if ( v16 == v17 )
      {
LABEL_28:
        v15 = v46;
        goto LABEL_29;
      }
    }
  }
  v39 = 0;
LABEL_29:
  if ( v15 != (__int64 *)v48 )
    _libc_free((unsigned __int64)v15);
  if ( !v53 )
    _libc_free((unsigned __int64)v50);
  return v39;
}
