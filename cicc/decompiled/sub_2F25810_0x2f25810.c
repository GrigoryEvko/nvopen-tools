// Function: sub_2F25810
// Address: 0x2f25810
//
__int64 __fastcall sub_2F25810(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rdi
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 v18; // [rsp+10h] [rbp-A0h] BYREF
  char v19; // [rsp+90h] [rbp-20h] BYREF

  v7 = *(_QWORD *)(sub_2EB2140(a4, &qword_50209B8, (__int64)a3) + 8);
  v8 = *(_QWORD *)(*a3 + 40);
  v9 = *(unsigned int *)(v7 + 88);
  v10 = *(_QWORD *)(v7 + 72);
  if ( !(_DWORD)v9 )
    goto LABEL_13;
  v11 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_50208B8 >> 9) ^ ((unsigned int)&unk_50208B8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; i = (v9 - 1) & v14 )
  {
    v13 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_50208B8 && v8 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_13;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v10 + 24 * v9 || (v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL)) == 0 )
LABEL_13:
    BUG();
  v16 = &v18;
  do
  {
    *v16 = -4096;
    v16 += 2;
  }
  while ( v16 != (__int64 *)&v19 );
  sub_2F25120(*a2, *(_QWORD *)(v15 + 8), a3);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
