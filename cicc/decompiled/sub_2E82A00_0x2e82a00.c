// Function: sub_2E82A00
// Address: 0x2e82a00
//
_QWORD *__fastcall sub_2E82A00(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // r13
  __int64 (*v13)(); // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  int v19; // r10d
  unsigned int i; // eax
  __int64 v21; // r8
  unsigned int v22; // eax
  __int64 v23; // rcx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  _QWORD *v27; // r13
  void (*v28)(); // rax
  __int64 v30; // rax
  int v31; // [rsp+Ch] [rbp-D4h]
  __int64 v32; // [rsp+10h] [rbp-D0h]
  __int64 v34; // [rsp+18h] [rbp-C8h]
  __int64 v35; // [rsp+18h] [rbp-C8h]
  __int64 v36; // [rsp+30h] [rbp-B0h] BYREF
  char v37; // [rsp+B0h] [rbp-30h] BYREF

  v5 = 0;
  v8 = sub_B2BE50(a3);
  v11 = a4;
  v12 = (__int64 *)v8;
  v13 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
  if ( v13 != sub_23CE270 )
  {
    v30 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v13)(*a2, a3, v9, v10, a4);
    v11 = a4;
    v5 = v30;
  }
  v14 = sub_BC1CD0(v11, &unk_4F82410, a3);
  v15 = *(_QWORD *)(a3 + 40);
  v16 = *(_QWORD *)(v14 + 8);
  v17 = *(unsigned int *)(v16 + 88);
  v18 = *(_QWORD *)(v16 + 72);
  if ( !(_DWORD)v17 )
    goto LABEL_21;
  v19 = 1;
  for ( i = (v17 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_50208B8 >> 9) ^ ((unsigned int)&unk_50208B8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)))); ; i = (v17 - 1) & v22 )
  {
    v21 = v18 + 24LL * i;
    if ( *(_UNKNOWN **)v21 == &unk_50208B8 && v15 == *(_QWORD *)(v21 + 8) )
      break;
    if ( *(_QWORD *)v21 == -4096 && *(_QWORD *)(v21 + 8) == -4096 )
      goto LABEL_21;
    v22 = v19 + i;
    ++v19;
  }
  if ( v21 == v18 + 24 * v17 || (v23 = *(_QWORD *)(*(_QWORD *)(v21 + 16) + 24LL)) == 0 )
LABEL_21:
    BUG();
  v24 = &v36;
  do
  {
    *v24 = -4096;
    v24 += 2;
  }
  while ( v24 != (__int64 *)&v37 );
  v34 = *(_QWORD *)(v23 + 8);
  v31 = sub_B6FBA0(v12, a3);
  v25 = *(_QWORD *)(v34 + 2480);
  if ( !v25 )
    v25 = v34 + 8;
  v32 = v25;
  v35 = *a2;
  v26 = sub_22077B0(0x460u);
  v27 = (_QWORD *)v26;
  if ( v26 )
    sub_2E81B70(v26, a3, v35, v5, v32, v31);
  sub_2E78D90(v27, v5);
  v28 = *(void (**)())(*(_QWORD *)*a2 + 248LL);
  if ( v28 != nullsub_1497 )
    ((void (__fastcall *)(__int64, _QWORD *))v28)(*a2, v27);
  *a1 = v27;
  return a1;
}
