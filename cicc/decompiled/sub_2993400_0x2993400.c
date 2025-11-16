// Function: sub_2993400
// Address: 0x2993400
//
__int64 __fastcall sub_2993400(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rbx
  int v11; // eax
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // eax
  unsigned int v19; // esi
  __int64 v20; // rcx
  int v21; // r10d
  __int64 v22; // rbx
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 result; // rax
  int v29; // eax
  int v30; // eax
  __int64 v31; // rdi
  char **v32; // rdx
  __int64 v33; // rcx
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rsi
  __int64 v36; // rsi
  char **v37; // rcx
  _BYTE *v38; // rdi
  unsigned __int64 v39; // r15
  __int64 v40; // rdi
  __int64 v41; // [rsp+8h] [rbp-F8h]
  __int64 v42; // [rsp+10h] [rbp-F0h] BYREF
  int v43; // [rsp+18h] [rbp-E8h]
  char *v44; // [rsp+20h] [rbp-E0h]
  __int64 v45; // [rsp+28h] [rbp-D8h]
  char v46; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v47; // [rsp+70h] [rbp-90h] BYREF
  _BYTE *v48; // [rsp+78h] [rbp-88h]
  __int64 v49; // [rsp+80h] [rbp-80h]
  _BYTE v50[120]; // [rsp+88h] [rbp-78h] BYREF

  v6 = sub_AA5930(a3);
  v9 = v8;
  v10 = v6;
  while ( v9 != v10 )
  {
    v17 = sub_ACADE0(*(__int64 ***)(v10 + 8));
    v18 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    if ( v18 == *(_DWORD *)(v10 + 72) )
    {
      v41 = v17;
      sub_B48D90(v10);
      v17 = v41;
      v18 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    }
    v11 = (v18 + 1) & 0x7FFFFFF;
    v12 = v11 | *(_DWORD *)(v10 + 4) & 0xF8000000;
    v13 = *(_QWORD *)(v10 - 8) + 32LL * (unsigned int)(v11 - 1);
    *(_DWORD *)(v10 + 4) = v12;
    if ( *(_QWORD *)v13 )
    {
      v14 = *(_QWORD *)(v13 + 8);
      **(_QWORD **)(v13 + 16) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *(_QWORD *)(v13 + 16);
    }
    *(_QWORD *)v13 = v17;
    if ( v17 )
    {
      v15 = *(_QWORD *)(v17 + 16);
      *(_QWORD *)(v13 + 8) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = v13 + 8;
      *(_QWORD *)(v13 + 16) = v17 + 16;
      *(_QWORD *)(v17 + 16) = v13;
    }
    *(_QWORD *)(*(_QWORD *)(v10 - 8)
              + 32LL * *(unsigned int *)(v10 + 72)
              + 8LL * ((*(_DWORD *)(v10 + 4) & 0x7FFFFFFu) - 1)) = a2;
    v16 = *(_QWORD *)(v10 + 32);
    if ( !v16 )
      BUG();
    v10 = 0;
    if ( *(_BYTE *)(v16 - 24) == 84 )
      v10 = v16 - 24;
  }
  v19 = *(_DWORD *)(a1 + 600);
  v42 = a3;
  v43 = 0;
  if ( !v19 )
  {
    ++*(_QWORD *)(a1 + 576);
    v47 = 0;
LABEL_44:
    sub_B23080(a1 + 576, 2 * v19);
LABEL_45:
    sub_B1C700(a1 + 576, &v42, &v47);
    v31 = v42;
    v22 = v47;
    v32 = (char **)&v47;
    v30 = *(_DWORD *)(a1 + 592) + 1;
    goto LABEL_32;
  }
  v20 = *(_QWORD *)(a1 + 584);
  v21 = 1;
  v22 = 0;
  v23 = (v19 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v24 = v20 + 16LL * v23;
  v25 = *(_QWORD *)v24;
  if ( *(_QWORD *)v24 == a3 )
  {
LABEL_18:
    v26 = *(unsigned int *)(v24 + 8);
    goto LABEL_19;
  }
  while ( v25 != -4096 )
  {
    if ( !v22 && v25 == -8192 )
      v22 = v24;
    v7 = (unsigned int)(v21 + 1);
    v23 = (v19 - 1) & (v21 + v23);
    v24 = v20 + 16LL * v23;
    v25 = *(_QWORD *)v24;
    if ( *(_QWORD *)v24 == a3 )
      goto LABEL_18;
    ++v21;
  }
  if ( !v22 )
    v22 = v24;
  v29 = *(_DWORD *)(a1 + 592);
  ++*(_QWORD *)(a1 + 576);
  v30 = v29 + 1;
  v47 = v22;
  if ( 4 * v30 >= 3 * v19 )
    goto LABEL_44;
  v31 = a3;
  v25 = v19 >> 3;
  v32 = (char **)&v47;
  if ( v19 - *(_DWORD *)(a1 + 596) - v30 <= (unsigned int)v25 )
  {
    sub_B23080(a1 + 576, v19);
    goto LABEL_45;
  }
LABEL_32:
  *(_DWORD *)(a1 + 592) = v30;
  if ( *(_QWORD *)v22 != -4096 )
    --*(_DWORD *)(a1 + 596);
  *(_QWORD *)v22 = v31;
  *(_DWORD *)(v22 + 8) = v43;
  v33 = *(unsigned int *)(a1 + 616);
  v34 = *(unsigned int *)(a1 + 620);
  v44 = &v46;
  v35 = v33 + 1;
  v45 = 0x800000000LL;
  v49 = 0x800000000LL;
  v26 = v33;
  v47 = a3;
  v48 = v50;
  if ( v33 + 1 > v34 )
  {
    v39 = *(_QWORD *)(a1 + 608);
    v40 = a1 + 608;
    if ( v39 > (unsigned __int64)&v47 || (unsigned __int64)&v47 >= v39 + 88 * v33 )
    {
      sub_298C980(v40, v35, (__int64)&v47, v33, v25, v7);
      v33 = *(unsigned int *)(a1 + 616);
      v36 = *(_QWORD *)(a1 + 608);
      v32 = (char **)&v47;
      v26 = v33;
    }
    else
    {
      sub_298C980(v40, v35, (__int64)&v47, v33, v25, v7);
      v36 = *(_QWORD *)(a1 + 608);
      v33 = *(unsigned int *)(a1 + 616);
      v32 = (char **)((char *)&v47 + v36 - v39);
      v26 = v33;
    }
  }
  else
  {
    v36 = *(_QWORD *)(a1 + 608);
  }
  v37 = (char **)(v36 + 88 * v33);
  if ( v37 )
  {
    *v37 = *v32;
    v37[1] = (char *)(v37 + 3);
    v37[2] = (char *)0x800000000LL;
    if ( *((_DWORD *)v32 + 4) )
      sub_2988A20((__int64)(v37 + 1), v32 + 1, (__int64)v32, (__int64)v37, v25, v7);
    v26 = *(unsigned int *)(a1 + 616);
  }
  v38 = v48;
  *(_DWORD *)(a1 + 616) = v26 + 1;
  if ( v38 != v50 )
  {
    _libc_free((unsigned __int64)v38);
    v26 = (unsigned int)(*(_DWORD *)(a1 + 616) - 1);
  }
  *(_DWORD *)(v22 + 8) = v26;
LABEL_19:
  v27 = *(_QWORD *)(a1 + 608) + 88 * v26;
  result = *(unsigned int *)(v27 + 16);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v27 + 20) )
  {
    sub_C8D5F0(v27 + 8, (const void *)(v27 + 24), result + 1, 8u, v25, v7);
    result = *(unsigned int *)(v27 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(v27 + 8) + 8 * result) = a2;
  ++*(_DWORD *)(v27 + 16);
  return result;
}
