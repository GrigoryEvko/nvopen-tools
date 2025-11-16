// Function: sub_2FB2980
// Address: 0x2fb2980
//
void __fastcall sub_2FB2980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v10; // r15
  int v11; // r10d
  unsigned __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // r14
  _QWORD *v28; // rdx
  _QWORD *v29; // rdi
  __int64 v30; // [rsp+0h] [rbp-B0h]
  int v31; // [rsp+Ch] [rbp-A4h]
  __int64 v32; // [rsp+20h] [rbp-90h]
  int *v33; // [rsp+28h] [rbp-88h]
  _BYTE *v34; // [rsp+30h] [rbp-80h] BYREF
  __int64 v35; // [rsp+38h] [rbp-78h]
  _BYTE v36[112]; // [rsp+40h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a1 + 72);
  v34 = v36;
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(unsigned int *)(v6 + 64);
  v35 = 0x800000000LL;
  v32 = *(_QWORD *)v7 + 4LL * *(unsigned int *)(v7 + 8);
  if ( v32 == *(_QWORD *)v7 + 4 * v8 )
    return;
  v33 = (int *)(*(_QWORD *)v7 + 4 * v8);
  do
  {
    v10 = *(_QWORD *)(a1 + 8);
    v11 = *v33;
    v12 = *(unsigned int *)(v10 + 160);
    v13 = *v33 & 0x7FFFFFFF;
    v14 = 8LL * v13;
    if ( v13 >= (unsigned int)v12 || (v15 = *(_QWORD *)(*(_QWORD *)(v10 + 152) + 8LL * v13)) == 0 )
    {
      v22 = v13 + 1;
      if ( (unsigned int)v12 < v22 )
      {
        v25 = v22;
        if ( v22 != v12 )
        {
          if ( v22 >= v12 )
          {
            v26 = *(_QWORD *)(v10 + 168);
            v27 = v25 - v12;
            if ( v25 > *(unsigned int *)(v10 + 164) )
            {
              v30 = *(_QWORD *)(v10 + 168);
              v31 = *v33;
              sub_C8D5F0(v10 + 152, (const void *)(v10 + 168), v25, 8u, a5, a6);
              v12 = *(unsigned int *)(v10 + 160);
              v26 = v30;
              v11 = v31;
            }
            v23 = *(_QWORD *)(v10 + 152);
            v28 = (_QWORD *)(v23 + 8 * v12);
            v29 = &v28[v27];
            if ( v28 != v29 )
            {
              do
                *v28++ = v26;
              while ( v29 != v28 );
              LODWORD(v12) = *(_DWORD *)(v10 + 160);
              v23 = *(_QWORD *)(v10 + 152);
            }
            *(_DWORD *)(v10 + 160) = v27 + v12;
            goto LABEL_21;
          }
          *(_DWORD *)(v10 + 160) = v22;
        }
      }
      v23 = *(_QWORD *)(v10 + 152);
LABEL_21:
      v24 = sub_2E10F30(v11);
      *(_QWORD *)(v23 + v14) = v24;
      v15 = v24;
      sub_2E11E80((_QWORD *)v10, v24);
    }
    v16 = *(_QWORD *)v15;
    v17 = *(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8);
    if ( v17 != *(_QWORD *)v15 )
    {
      do
      {
        v18 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 8LL);
        if ( *(_QWORD *)(v16 + 8) == (v18 & 0xFFFFFFFFFFFFFFF8LL | 6) && (v18 & 6) != 0 )
        {
          v19 = *(_QWORD *)((*(_QWORD *)(*(_QWORD *)(v16 + 16) + 8LL) & 0xFFFFFFFFFFFFFFF8LL) + 16);
          sub_2E8F690(v19, *(_DWORD *)(v15 + 112), *(_QWORD **)(a1 + 48), 0);
          if ( (unsigned __int8)sub_2E8B940(v19) )
          {
            v20 = (unsigned int)v35;
            v21 = (unsigned int)v35 + 1LL;
            if ( v21 > HIDWORD(v35) )
            {
              sub_C8D5F0((__int64)&v34, v36, v21, 8u, a5, a6);
              v20 = (unsigned int)v35;
            }
            *(_QWORD *)&v34[8 * v20] = v19;
            LODWORD(v35) = v35 + 1;
          }
        }
        v16 += 24;
      }
      while ( v16 != v17 );
    }
    ++v33;
  }
  while ( (int *)v32 != v33 );
  if ( (_DWORD)v35 )
    sub_350D230(*(_QWORD *)(a1 + 72), &v34, 0, 0);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
}
