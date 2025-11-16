// Function: sub_2100320
// Address: 0x2100320
//
__int64 __fastcall sub_2100320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  _QWORD *v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r8
  __int16 v15; // si
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r9
  unsigned int v20; // edi
  __int64 *v21; // rdx
  __int64 v22; // r11
  __int64 v23; // r15
  __int64 v24; // rdi
  __int64 (__fastcall *v25)(__int64, __int64); // rax
  __int64 v26; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  int v30; // edx
  _QWORD *v31; // rdx
  _QWORD *v32; // [rsp+8h] [rbp-38h]
  int v33; // [rsp+8h] [rbp-38h]

  v10 = *(_QWORD **)(a1 + 96);
  v11 = *(_QWORD **)(a1 + 88);
  if ( v10 == v11 )
  {
    v12 = &v11[*(unsigned int *)(a1 + 108)];
    if ( v11 == v12 )
    {
      v31 = *(_QWORD **)(a1 + 88);
    }
    else
    {
      do
      {
        if ( a3 == *v11 )
          break;
        ++v11;
      }
      while ( v12 != v11 );
      v31 = v12;
    }
  }
  else
  {
    v32 = &v10[*(unsigned int *)(a1 + 104)];
    v11 = sub_16CC9F0(a1 + 80, a3);
    v12 = v32;
    if ( a3 == *v11 )
    {
      v28 = *(_QWORD *)(a1 + 96);
      if ( v28 == *(_QWORD *)(a1 + 88) )
        v29 = *(unsigned int *)(a1 + 108);
      else
        v29 = *(unsigned int *)(a1 + 104);
      v31 = (_QWORD *)(v28 + 8 * v29);
    }
    else
    {
      v13 = *(_QWORD *)(a1 + 96);
      if ( v13 != *(_QWORD *)(a1 + 88) )
      {
        v11 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a1 + 104));
        goto LABEL_5;
      }
      v11 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a1 + 108));
      v31 = v11;
    }
  }
  while ( v31 != v11 && *v11 >= 0xFFFFFFFFFFFFFFFELL )
    ++v11;
LABEL_5:
  if ( v12 == v11 )
    return 0;
  v14 = *(_QWORD *)(a2 + 8);
  v15 = *(_WORD *)(v14 + 46);
  v16 = v14;
  if ( (v15 & 4) != 0 )
  {
    do
      v16 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v16 + 46) & 4) != 0 );
  }
  v17 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
  v18 = *(unsigned int *)(v17 + 384);
  v19 = *(_QWORD *)(v17 + 368);
  if ( !(_DWORD)v18 )
    goto LABEL_30;
  v20 = (v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v21 = (__int64 *)(v19 + 16LL * v20);
  v22 = *v21;
  if ( *v21 != v16 )
  {
    v30 = 1;
    while ( v22 != -8 )
    {
      v20 = (v18 - 1) & (v30 + v20);
      v33 = v30 + 1;
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == v16 )
        goto LABEL_10;
      v30 = v33;
    }
LABEL_30:
    v21 = (__int64 *)(v19 + 16 * v18);
  }
LABEL_10:
  v23 = v21[1];
  if ( !a5 )
    return sub_2100010(a1, v14, v23, a4, v14, v19);
  v24 = *(_QWORD *)(a1 + 48);
  v25 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v24 + 136LL);
  if ( v25 == sub_1DF74E0 )
  {
    if ( (v15 & 4) != 0 || (v15 & 8) == 0 )
      v26 = (*(_QWORD *)(*(_QWORD *)(v14 + 16) + 8LL) >> 26) & 1LL;
    else
      LOBYTE(v26) = sub_1E15D00(v14, 0x4000000u, 2);
  }
  else
  {
    LOBYTE(v26) = v25(v24, v14);
  }
  if ( (_BYTE)v26 )
  {
    v14 = *(_QWORD *)(a2 + 8);
    return sub_2100010(a1, v14, v23, a4, v14, v19);
  }
  return 0;
}
