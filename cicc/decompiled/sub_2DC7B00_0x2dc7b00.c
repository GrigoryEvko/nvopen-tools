// Function: sub_2DC7B00
// Address: 0x2dc7b00
//
__int64 __fastcall sub_2DC7B00(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 (*v5)(); // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  int v16; // r11d
  unsigned int i; // eax
  __int64 v18; // r9
  unsigned int v19; // eax
  __int64 *v20; // r9
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rsi
  int v25; // r10d
  unsigned int j; // eax
  __int64 v27; // rdi
  unsigned int v28; // eax
  unsigned __int64 v29; // rax
  __int64 *v31; // [rsp+8h] [rbp-C8h]
  __int64 v32; // [rsp+20h] [rbp-B0h] BYREF
  char v33; // [rsp+A0h] [rbp-30h] BYREF

  v5 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
  if ( v5 == sub_23CE270 )
    BUG();
  v8 = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(*a2, a3);
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 144LL);
  if ( v9 != sub_2C8F680 )
    ((void (__fastcall *)(__int64))v9)(v8);
  v10 = (__int64 *)(sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8);
  v31 = (__int64 *)(sub_BC1CD0(a4, &unk_4F89C30, a3) + 8);
  v11 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v12 = *(_QWORD *)(a3 + 40);
  v13 = *(_QWORD *)(v11 + 8);
  v14 = *(unsigned int *)(v13 + 88);
  v15 = *(_QWORD *)(v13 + 72);
  if ( !(_DWORD)v14 )
    goto LABEL_29;
  v16 = 1;
  for ( i = (v14 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)))); ; i = (v14 - 1) & v19 )
  {
    v18 = v15 + 24LL * i;
    if ( *(_UNKNOWN **)v18 == &unk_4F87C68 && v12 == *(_QWORD *)(v18 + 8) )
      break;
    if ( *(_QWORD *)v18 == -4096 && *(_QWORD *)(v18 + 8) == -4096 )
      goto LABEL_29;
    v19 = v16 + i;
    ++v16;
  }
  if ( v18 == v15 + 24 * v14 )
  {
LABEL_29:
    v21 = 0;
    v20 = 0;
  }
  else
  {
    v20 = *(__int64 **)(*(_QWORD *)(v18 + 16) + 24LL);
    if ( v20 )
    {
      v21 = (__int64)(v20 + 1);
      v22 = &v32;
      do
      {
        *v22 = -4096;
        v22 += 2;
      }
      while ( v22 != (__int64 *)&v33 );
      v20 = (__int64 *)v20[2];
      if ( v20 )
        v20 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8);
    }
    else
    {
      v21 = 0;
    }
  }
  v23 = *(unsigned int *)(a4 + 88);
  v24 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v23 )
    goto LABEL_27;
  v25 = 1;
  for ( j = (v23 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = (v23 - 1) & v28 )
  {
    v27 = v24 + 24LL * j;
    if ( *(_UNKNOWN **)v27 == &unk_4F81450 && a3 == *(_QWORD *)(v27 + 8) )
      break;
    if ( *(_QWORD *)v27 == -4096 && *(_QWORD *)(v27 + 8) == -4096 )
      goto LABEL_27;
    v28 = v25 + j;
    ++v25;
  }
  if ( v27 == v24 + 24 * v23 )
  {
LABEL_27:
    v29 = 0;
  }
  else
  {
    v29 = *(_QWORD *)(*(_QWORD *)(v27 + 16) + 24LL);
    if ( v29 )
      v29 += 8LL;
  }
  sub_2DC4260(a1, a3, v10, v31, v21, v20, v29);
  return a1;
}
