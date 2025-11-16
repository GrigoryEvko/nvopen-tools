// Function: sub_36D0EF0
// Address: 0x36d0ef0
//
__int64 __fastcall sub_36D0EF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 i; // rax
  __int64 *v10; // rbx
  __int64 *v11; // r12
  __int64 v12; // rdi
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rax
  _BYTE *v18; // r13
  int v19; // ebx
  size_t v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rbx
  unsigned int v25; // eax
  __int64 *v26; // r12
  __int64 v27; // rbx
  __int64 *v28; // r13
  __int64 v29; // r14
  __int64 v30; // r15
  unsigned int v31; // eax
  __int64 v32; // r15
  unsigned int v33; // eax
  unsigned __int8 v34; // [rsp+18h] [rbp-1F8h]
  __int64 v35; // [rsp+18h] [rbp-1F8h]
  __int64 v36; // [rsp+20h] [rbp-1F0h]
  __int64 v37; // [rsp+28h] [rbp-1E8h]
  __int64 *v38; // [rsp+30h] [rbp-1E0h] BYREF
  __int64 v39; // [rsp+38h] [rbp-1D8h]
  _BYTE v40[128]; // [rsp+40h] [rbp-1D0h] BYREF
  void *src; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v42; // [rsp+C8h] [rbp-148h]
  _BYTE v43[128]; // [rsp+D0h] [rbp-140h] BYREF
  __int64 *v44; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+158h] [rbp-B8h]
  _BYTE v46[176]; // [rsp+160h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a2 + 328);
  v38 = (__int64 *)v40;
  v7 = *(_QWORD *)(a2 + 32);
  v39 = 0x1000000000LL;
  v8 = *(_QWORD *)(v6 + 56);
  v37 = v6 + 48;
  v34 = 0;
  if ( v6 + 48 != v8 )
  {
    while ( 1 )
    {
      if ( !v8 || (i = v8, (*(_BYTE *)v8 & 4) == 0) )
      {
        for ( i = v8; (*(_BYTE *)(i + 44) & 8) != 0; i = *(_QWORD *)(i + 8) )
          ;
      }
      v36 = *(_QWORD *)(i + 8);
      if ( (unsigned int)*(unsigned __int16 *)(v8 + 68) - 2907 > 1 )
        goto LABEL_7;
      src = v43;
      v42 = 0x1000000000LL;
      v45 = 0x1000000000LL;
      v14 = *(unsigned int *)(*(_QWORD *)(v8 + 32) + 8LL);
      v44 = (__int64 *)v46;
      if ( (int)v14 < 0 )
        v15 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 16 * (v14 & 0x7FFFFFFF) + 8);
      else
        v15 = *(_QWORD *)(*(_QWORD *)(v7 + 304) + 8 * v14);
      if ( !v15 )
        goto LABEL_38;
      if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
        break;
LABEL_18:
      v16 = *(_QWORD *)(v15 + 16);
LABEL_19:
      if ( (unsigned __int8)sub_36D0D50(v16, v7, (__int64)&src, (__int64)&v44, a5, a6) )
      {
        v17 = *(_QWORD *)(v15 + 16);
        while ( 1 )
        {
          v15 = *(_QWORD *)(v15 + 32);
          if ( !v15 )
            break;
          if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
          {
            v16 = *(_QWORD *)(v15 + 16);
            if ( v17 != v16 )
              goto LABEL_19;
          }
        }
        v18 = src;
        v19 = v42;
        v20 = 8LL * (unsigned int)v42;
        v21 = (unsigned int)v39;
        v22 = (unsigned int)v39 + (unsigned __int64)(unsigned int)v42;
        if ( v22 <= HIDWORD(v39) )
        {
LABEL_24:
          if ( v20 )
          {
            memcpy(&v38[v21], v18, v20);
            LODWORD(v21) = v39;
          }
          goto LABEL_26;
        }
        goto LABEL_45;
      }
      v28 = v44;
LABEL_32:
      if ( v28 != (__int64 *)v46 )
        _libc_free((unsigned __int64)v28);
      if ( src != v43 )
        _libc_free((unsigned __int64)src);
LABEL_7:
      v8 = v36;
      if ( v37 == v36 )
        goto LABEL_8;
    }
    while ( 1 )
    {
      v15 = *(_QWORD *)(v15 + 32);
      if ( !v15 )
        break;
      if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
        goto LABEL_18;
    }
LABEL_38:
    v22 = (unsigned int)v39;
    LODWORD(v21) = v39;
    if ( HIDWORD(v39) >= (unsigned int)v39 )
    {
      v19 = 0;
LABEL_26:
      LODWORD(v39) = v19 + v21;
      v23 = (unsigned int)(v19 + v21);
      if ( v23 + 1 > (unsigned __int64)HIDWORD(v39) )
      {
        sub_C8D5F0((__int64)&v38, v40, v23 + 1, 8u, a5, a6);
        v23 = (unsigned int)v39;
      }
      v38[v23] = v8;
      v24 = *(_QWORD *)(v8 + 32);
      LODWORD(v39) = v39 + 1;
      v25 = sub_2E88FE0(v8);
      v26 = v44;
      v27 = v24 + 40LL * v25;
      v28 = &v44[(unsigned int)v45];
      if ( v44 == v28 )
      {
        v34 = 1;
      }
      else
      {
        v35 = v7;
        do
        {
          v29 = *v26++;
          v30 = *(_QWORD *)(v29 + 32);
          v31 = sub_2E88FE0(v29);
          sub_2EAB400(v30 + 40LL * v31 + 240, *(_QWORD *)(v27 + 24), 0);
          v32 = *(_QWORD *)(v29 + 32);
          v33 = sub_2E88FE0(v29);
          sub_2EAB3B0(v32 + 40LL * v33 + 80, 101, 0);
        }
        while ( v28 != v26 );
        v7 = v35;
        v34 = 1;
        v28 = v44;
      }
      goto LABEL_32;
    }
    v18 = v43;
    v19 = 0;
    v20 = 0;
LABEL_45:
    sub_C8D5F0((__int64)&v38, v40, v22, 8u, a5, a6);
    v21 = (unsigned int)v39;
    goto LABEL_24;
  }
LABEL_8:
  v10 = v38;
  v11 = &v38[(unsigned int)v39];
  if ( v38 != v11 )
  {
    do
    {
      v12 = *v10++;
      sub_2E88E20(v12);
    }
    while ( v11 != v10 );
    v11 = v38;
  }
  if ( v11 != (__int64 *)v40 )
    _libc_free((unsigned __int64)v11);
  return v34;
}
