// Function: sub_1DAC6C0
// Address: 0x1dac6c0
//
__int64 __fastcall sub_1DAC6C0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned int v10; // r14d
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned int v16; // r14d
  __int64 v17; // r15
  __int64 v18; // r8
  unsigned int v19; // r9d
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // ecx
  unsigned int v23; // edi
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 *v30; // rcx
  __int64 v31; // rax
  _QWORD *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r14
  __int64 v35; // rdx
  unsigned int i; // esi
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r15
  unsigned __int8 v43; // [rsp+0h] [rbp-40h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  int v45; // [rsp+8h] [rbp-38h]
  unsigned __int8 v46; // [rsp+8h] [rbp-38h]
  int v47; // [rsp+8h] [rbp-38h]

  v4 = a1 + 8;
  if ( a2 != 1 )
  {
    v16 = a2;
    v19 = 0;
    v18 = a2 - 1;
    v17 = 16 * v18;
LABEL_6:
    if ( !*(_DWORD *)(a1 + 16) || (v20 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v20 + 12) >= *(_DWORD *)(v20 + 8)) )
    {
      v43 = v19;
      v45 = v18;
      sub_3945E40(v4, (unsigned int)v18);
      v19 = v43;
      LODWORD(v18) = v45;
      ++*(_DWORD *)(v17 + *(_QWORD *)(a1 + 8) + 12);
      v20 = *(_QWORD *)(a1 + 8);
    }
    v21 = v17 + v20;
    v22 = *(_DWORD *)(v21 + 8);
    if ( v22 == 12 )
    {
      v47 = v18;
      v19 = sub_1DAC330((_QWORD *)a1, v18);
      v42 = v47 + (unsigned int)(unsigned __int8)v19;
      LODWORD(v18) = v42;
      v16 = v42 + 1;
      v17 = 16 * v42;
      v21 = v17 + *(_QWORD *)(a1 + 8);
      v22 = *(_DWORD *)(v21 + 8);
    }
    v23 = *(_DWORD *)(v21 + 12);
    v24 = *(_QWORD *)v21;
    v25 = v22 - 1;
    if ( v22 != v23 )
    {
      do
      {
        v26 = v25 + 1;
        *(_QWORD *)(v24 + 8 * v26) = *(_QWORD *)(v24 + 8LL * v25);
        *(_QWORD *)(v24 + 8 * v26 + 96) = *(_QWORD *)(v24 + 8LL * v25 + 96);
        LODWORD(v26) = v25--;
      }
      while ( v23 != (_DWORD)v26 );
    }
    *(_QWORD *)(v24 + 8LL * v23) = a3;
    *(_QWORD *)(v24 + 8LL * v23 + 96) = a4;
    v27 = v17 + *(_QWORD *)(a1 + 8);
    v28 = *(unsigned int *)(v27 + 8);
    *(_DWORD *)(v27 + 8) = v28 + 1;
    if ( (_DWORD)v18 )
    {
      v29 = *(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v18 - 1);
      v30 = (unsigned __int64 *)(*(_QWORD *)v29 + 8LL * *(unsigned int *)(v29 + 12));
      *v30 = *v30 & 0xFFFFFFFFFFFFFFC0LL | v28;
    }
    v31 = *(_QWORD *)(a1 + 8);
    v32 = (_QWORD *)(v31 + v17);
    v33 = *(unsigned int *)(v31 + v17 + 12);
    if ( (_DWORD)v33 == *(_DWORD *)(v31 + v17 + 8) - 1 )
    {
      v46 = v19;
      sub_1DA99F0(a1, v18, a4);
      v31 = *(_QWORD *)(a1 + 8);
      v19 = v46;
      v32 = (_QWORD *)(v31 + v17);
      v33 = *(unsigned int *)(v31 + v17 + 12);
    }
    v34 = v31 + 16LL * v16;
    v35 = *(_QWORD *)(*v32 + 8 * v33);
    *(_QWORD *)v34 = v35 & 0xFFFFFFFFFFFFFFC0LL;
    *(_DWORD *)(v34 + 8) = (v35 & 0x3F) + 1;
    return v19;
  }
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_DWORD *)(*(_QWORD *)a1 + 84LL);
  if ( v10 > 3 )
  {
    LODWORD(v44) = *(_DWORD *)(v9 + 12);
    v11 = (unsigned __int64)sub_1DA8890(*(_QWORD *)(v8 + 88));
    v12 = 0;
    do
    {
      *(_QWORD *)(v11 + 8 * v12) = *(_QWORD *)(v8 + 8 * v12 + 8);
      *(_QWORD *)(v11 + 8 * v12 + 96) = *(_QWORD *)(v8 + 8 * v12 + 40);
      ++v12;
    }
    while ( v10 != (_DWORD)v12 );
    v13 = v11 & 0xFFFFFFFFFFFFFFC0LL;
    v14 = v8 + 8;
    v15 = *(_QWORD *)(v13 + 120);
    ++*(_DWORD *)(v8 + 80);
    v16 = 2;
    *(_QWORD *)(v8 + 8) = v13 | 3;
    *(_QWORD *)(v8 + 40) = v15;
    *(_DWORD *)(v8 + 84) = 1;
    v17 = 16;
    sub_3945C20(a1 + 8, v14, 1, v44 << 32);
    v4 = a1 + 8;
    LODWORD(v18) = 1;
    v19 = 1;
    goto LABEL_6;
  }
  for ( i = *(_DWORD *)(v9 + 12); i != v10; *(_QWORD *)(v8 + 8 * v38 + 40) = *(_QWORD *)(v8 + 8LL * v10 + 40) )
  {
    v38 = v10--;
    *(_QWORD *)(v8 + 8 * v38 + 8) = *(_QWORD *)(v8 + 8LL * v10 + 8);
  }
  v19 = 0;
  *(_QWORD *)(v8 + 8LL * i + 8) = a3;
  *(_QWORD *)(v8 + 8LL * i + 40) = a4;
  v39 = *(_DWORD *)(v8 + 84) + 1;
  *(_DWORD *)(v8 + 84) = v39;
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v39;
  v40 = *(_QWORD *)(a1 + 8);
  v41 = *(_QWORD *)(*(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 12));
  *(_QWORD *)(v40 + 16) = v41 & 0xFFFFFFFFFFFFFFC0LL;
  *(_DWORD *)(v40 + 24) = (v41 & 0x3F) + 1;
  return v19;
}
