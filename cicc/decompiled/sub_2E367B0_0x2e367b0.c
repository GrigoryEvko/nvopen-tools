// Function: sub_2E367B0
// Address: 0x2e367b0
//
__int64 __fastcall sub_2E367B0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // esi
  unsigned int v22; // r10d
  int v23; // r9d
  unsigned int i; // eax
  __int64 v25; // r8
  unsigned int v26; // eax
  int v27; // r11d
  unsigned int j; // eax
  __int64 v29; // rdi
  unsigned int v30; // eax
  int v31; // r11d
  unsigned int k; // eax
  __int64 v33; // rdi
  unsigned int v34; // eax
  int v35; // r11d
  unsigned int m; // eax
  __int64 v37; // rsi
  unsigned int v38; // eax
  __int64 v39; // [rsp+0h] [rbp-70h]
  __int64 v43[10]; // [rsp+20h] [rbp-50h] BYREF

  v6 = a3;
  if ( a3 )
  {
    v8 = sub_B82360(a3[1], (__int64)&unk_501EACC);
    if ( v8 && (v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_501EACC)) != 0 )
      v10 = v9 + 200;
    else
      v10 = 0;
    v11 = sub_B82360(v6[1], (__int64)&unk_5025C1C);
    if ( v11 && (v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_5025C1C)) != 0 )
      v13 = v12 + 200;
    else
      v13 = 0;
    v14 = sub_B82360(v6[1], (__int64)&unk_501EB14);
    if ( v14 && (v15 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_501EB14)) != 0 )
      v16 = v15 + 200;
    else
      v16 = 0;
    v6 = (__int64 *)sub_B82360(v6[1], (__int64)&unk_50208AC);
    if ( v6 )
    {
      v6 = (__int64 *)(*(__int64 (__fastcall **)(__int64 *, void *))(*v6 + 104))(v6, &unk_50208AC);
      if ( v6 )
        v6 += 25;
    }
    goto LABEL_14;
  }
  v19 = *(_QWORD *)(a1 + 32);
  v20 = *(_QWORD *)(a4 + 72);
  v21 = *(_DWORD *)(a4 + 88);
  if ( !v21 )
  {
LABEL_32:
    v39 = v20 + 24LL * v21;
    if ( !v21 )
    {
      v10 = 0;
      v13 = 0;
      v16 = 0;
      goto LABEL_14;
    }
    v22 = v21 - 1;
    goto LABEL_62;
  }
  v22 = v21 - 1;
  v23 = 1;
  for ( i = (v21 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_501EAD0 >> 9) ^ ((unsigned int)&unk_501EAD0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; i = v22 & v26 )
  {
    v25 = v20 + 24LL * i;
    if ( *(_UNKNOWN **)v25 == &unk_501EAD0 && v19 == *(_QWORD *)(v25 + 8) )
      break;
    if ( *(_QWORD *)v25 == -4096 && *(_QWORD *)(v25 + 8) == -4096 )
      goto LABEL_32;
    v26 = v23 + i;
    ++v23;
  }
  v39 = v20 + 24LL * v21;
  if ( v39 == v25 )
  {
LABEL_62:
    v10 = 0;
    goto LABEL_24;
  }
  v10 = *(_QWORD *)(*(_QWORD *)(v25 + 16) + 24LL);
  if ( v10 )
    v10 += 8;
LABEL_24:
  v27 = 1;
  for ( j = v22
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_5025C20 >> 9) ^ ((unsigned int)&unk_5025C20 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; j = v22 & v30 )
  {
    v29 = v20 + 24LL * j;
    if ( *(_UNKNOWN **)v29 == &unk_5025C20 && v19 == *(_QWORD *)(v29 + 8) )
      break;
    if ( *(_QWORD *)v29 == -4096 && *(_QWORD *)(v29 + 8) == -4096 )
      goto LABEL_58;
    v30 = v27 + j;
    ++v27;
  }
  if ( v39 == v29 )
  {
LABEL_58:
    v13 = 0;
    goto LABEL_38;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(v29 + 16) + 24LL);
  if ( v13 )
    v13 += 8;
LABEL_38:
  v31 = 1;
  for ( k = v22
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_501EB18 >> 9) ^ ((unsigned int)&unk_501EB18 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; k = v22 & v34 )
  {
    v33 = v20 + 24LL * k;
    if ( *(_UNKNOWN **)v33 == &unk_501EB18 && v19 == *(_QWORD *)(v33 + 8) )
      break;
    if ( *(_QWORD *)v33 == -4096 && *(_QWORD *)(v33 + 8) == -4096 )
      goto LABEL_56;
    v34 = v31 + k;
    ++v31;
  }
  if ( v39 == v33 )
  {
LABEL_56:
    v16 = 0;
    goto LABEL_46;
  }
  v16 = *(_QWORD *)(*(_QWORD *)(v33 + 16) + 24LL);
  if ( v16 )
    v16 += 8;
LABEL_46:
  v35 = 1;
  for ( m = v22
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_50208B0 >> 9) ^ ((unsigned int)&unk_50208B0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; m = v22 & v38 )
  {
    v37 = v20 + 24LL * m;
    if ( *(_UNKNOWN **)v37 == &unk_50208B0 && v19 == *(_QWORD *)(v37 + 8) )
      break;
    if ( *(_QWORD *)v37 == -4096 && *(_QWORD *)(v37 + 8) == -4096 )
      goto LABEL_14;
    v38 = v35 + m;
    ++v35;
  }
  if ( v39 != v37 )
  {
    v6 = *(__int64 **)(*(_QWORD *)(v37 + 16) + 24LL);
    if ( v6 )
      ++v6;
  }
LABEL_14:
  v43[0] = v10;
  v43[1] = v13;
  v43[2] = v16;
  v43[3] = (__int64)v6;
  return sub_2E350C0(a1, a2, v43, a5, a6);
}
