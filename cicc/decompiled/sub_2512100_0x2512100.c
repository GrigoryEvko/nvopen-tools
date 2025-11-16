// Function: sub_2512100
// Address: 0x2512100
//
__int64 __fastcall sub_2512100(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v3; // ecx
  int v6; // ecx
  __int64 *v7; // r13
  int v8; // r12d
  __int64 v9; // r8
  __int64 v10; // r11
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  unsigned int i; // eax
  __int64 *v18; // rdx
  __int64 v19; // r10
  unsigned int v20; // eax
  __int64 v22; // [rsp-8h] [rbp-8h]

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v3 - 1;
  v7 = 0;
  v8 = 1;
  v9 = a2[1];
  v10 = *(_QWORD *)(a1 + 8);
  v11 = a2[2];
  v12 = unk_4FEE4D0;
  v13 = unk_4FEE4D8;
  v14 = qword_4FEE4C0[0];
  *(&v22 - 6) = qword_4FEE4C0[1];
  v15 = *a2;
  v16 = 0xBF58476D1CE4E5B9LL
      * (((unsigned __int64)(((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)) << 32)
       | ((unsigned int)v11 >> 9)
       ^ ((unsigned int)v11 >> 4)
       ^ (16 * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4))));
  for ( i = v6 & ((v16 >> 31) ^ v16); ; i = v6 & v20 )
  {
    v18 = (__int64 *)(v10 + 32LL * i);
    v19 = *v18;
    if ( *v18 == v15 && v9 == v18[1] && v11 == v18[2] )
    {
      *a3 = v18;
      return 1;
    }
    if ( v19 == -4096 )
      break;
    if ( v19 == -8192 && v18[1] == v14 && v18[2] == *(&v22 - 6) && !v7 )
      v7 = (__int64 *)(v10 + 32LL * i);
LABEL_7:
    v20 = v8 + i;
    ++v8;
  }
  if ( v12 != v18[1] || v13 != v18[2] )
    goto LABEL_7;
  if ( !v7 )
    v7 = (__int64 *)(v10 + 32LL * i);
  *a3 = v7;
  return 0;
}
