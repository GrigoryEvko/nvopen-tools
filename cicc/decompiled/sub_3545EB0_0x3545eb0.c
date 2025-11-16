// Function: sub_3545EB0
// Address: 0x3545eb0
//
__int64 __fastcall sub_3545EB0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v10; // r8
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  _BYTE *v14; // rdi
  _QWORD *v15; // rbx
  __int64 v16; // rax
  bool v17; // zf
  __int64 v18; // r14
  __int64 *v19; // rcx
  __int64 *v20; // rsi
  __int64 v21; // r8
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // r10
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 i; // r10
  unsigned __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rax
  unsigned __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+10h] [rbp-90h]
  __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 v38; // [rsp+18h] [rbp-88h]
  _BYTE *v39; // [rsp+20h] [rbp-80h] BYREF
  __int64 v40; // [rsp+28h] [rbp-78h]
  _BYTE v41[112]; // [rsp+30h] [rbp-70h] BYREF

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 8;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  v7 = a3[6];
  v8 = a3[7];
  v39 = v41;
  v40 = 0x800000000LL;
  if ( v7 == v8 )
  {
    v14 = v41;
    LODWORD(v13) = 0;
  }
  else
  {
    do
    {
      while ( (*(_BYTE *)(v7 + 254) & 8) == 0
           || !*(_QWORD *)v7
           || !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a4 + 16LL))(a4) )
      {
        v7 += 256;
        if ( v8 == v7 )
          goto LABEL_10;
      }
      v11 = (unsigned int)v40;
      v12 = (unsigned int)v40 + 1LL;
      if ( v12 > HIDWORD(v40) )
      {
        sub_C8D5F0((__int64)&v39, v41, v12, 8u, v10, a6);
        v11 = (unsigned int)v40;
      }
      *(_QWORD *)&v39[8 * v11] = v7;
      v7 += 256;
      LODWORD(v40) = v40 + 1;
    }
    while ( v8 != v7 );
LABEL_10:
    LODWORD(v13) = v40;
    v14 = v39;
  }
  v15 = (_QWORD *)a3[433];
LABEL_12:
  if ( !(_DWORD)v13 )
    goto LABEL_33;
  do
  {
    v16 = (unsigned int)v13;
    v13 = (unsigned int)(v13 - 1);
    v17 = *(_BYTE *)(a1 + 28) == 0;
    v18 = *(_QWORD *)&v14[8 * v16 - 8];
    LODWORD(v40) = v13;
    if ( !v17 )
    {
      v19 = *(__int64 **)(a1 + 8);
      v20 = &v19[*(unsigned int *)(a1 + 20)];
      v21 = *(unsigned int *)(a1 + 20);
      v22 = v19;
      if ( v19 != v20 )
      {
        while ( v18 != *v22 )
        {
          if ( v20 == ++v22 )
          {
            if ( v18 == *v19 )
              goto LABEL_20;
            goto LABEL_18;
          }
        }
        goto LABEL_12;
      }
LABEL_40:
      if ( *(_DWORD *)(a1 + 16) > (unsigned int)v21 )
      {
        *(_DWORD *)(a1 + 20) = v21 + 1;
        *v22 = v18;
        ++*(_QWORD *)a1;
        goto LABEL_20;
      }
LABEL_42:
      sub_C8CC70(a1, v18, v13, (__int64)v19, v21, a6);
      goto LABEL_20;
    }
    if ( sub_C8CA60(a1, v18) )
      goto LABEL_37;
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_42;
    v19 = *(__int64 **)(a1 + 8);
    v21 = *(unsigned int *)(a1 + 20);
    v22 = &v19[v21];
    if ( v19 == v22 )
      goto LABEL_40;
    while ( v18 != *v19 )
    {
LABEL_18:
      if ( v22 == ++v19 )
        goto LABEL_40;
    }
LABEL_20:
    v23 = sub_35459D0(v15, v18);
    v24 = *(unsigned int *)(v23 + 8);
    v25 = *(_QWORD *)v23;
    v26 = (unsigned int)v40;
    for ( i = v25 + 32 * v24; v25 != i; LODWORD(v40) = v40 + 1 )
    {
      v28 = *(_QWORD *)(v25 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v26 + 1 > (unsigned __int64)HIDWORD(v40) )
      {
        v33 = *(_QWORD *)(v25 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        v34 = i;
        v37 = v25;
        sub_C8D5F0((__int64)&v39, v41, v26 + 1, 8u, v25, v28);
        v26 = (unsigned int)v40;
        v28 = v33;
        i = v34;
        v25 = v37;
      }
      v25 += 32;
      *(_QWORD *)&v39[8 * v26] = v28;
      v26 = (unsigned int)(v40 + 1);
    }
    v29 = sub_3545E90(v15, v18);
    v13 = (unsigned int)v40;
    v30 = *(_QWORD *)v29;
    v31 = *(_QWORD *)v29 + 32LL * *(unsigned int *)(v29 + 8);
    if ( v31 == v30 )
    {
LABEL_37:
      LODWORD(v13) = v40;
      v14 = v39;
      continue;
    }
    do
    {
      while ( *(_DWORD *)(v30 + 24) != 1 )
      {
        v30 += 32;
        if ( v31 == v30 )
          goto LABEL_31;
      }
      a6 = *(_QWORD *)v30;
      if ( v13 + 1 > (unsigned __int64)HIDWORD(v40) )
      {
        v35 = *(_QWORD *)v30;
        v38 = v31;
        sub_C8D5F0((__int64)&v39, v41, v13 + 1, 8u, v13 + 1, a6);
        v13 = (unsigned int)v40;
        a6 = v35;
        v31 = v38;
      }
      v30 += 32;
      *(_QWORD *)&v39[8 * v13] = a6;
      v13 = (unsigned int)(v40 + 1);
      LODWORD(v40) = v40 + 1;
    }
    while ( v31 != v30 );
LABEL_31:
    v14 = v39;
  }
  while ( (_DWORD)v13 );
LABEL_33:
  if ( v14 != v41 )
    _libc_free((unsigned __int64)v14);
  return a1;
}
