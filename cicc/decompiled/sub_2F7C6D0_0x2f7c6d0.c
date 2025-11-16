// Function: sub_2F7C6D0
// Address: 0x2f7c6d0
//
__int64 __fastcall sub_2F7C6D0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r10d
  unsigned int i; // eax
  __int64 v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 *v15; // rax
  __int64 v17; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 v19; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-98h]
  char v21; // [rsp+90h] [rbp-20h] BYREF

  v6 = *(_QWORD *)(*a3 + 40LL);
  v7 = *(_QWORD *)(sub_2EB2140(a4, &qword_50209B8, (__int64)a3) + 8);
  v8 = *(unsigned int *)(v7 + 88);
  v9 = *(_QWORD *)(v7 + 72);
  if ( !(_DWORD)v8 )
    goto LABEL_15;
  v10 = 1;
  for ( i = (v8 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_5024E68 >> 9) ^ ((unsigned int)&unk_5024E68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))); ; i = (v8 - 1) & v13 )
  {
    v12 = v9 + 24LL * i;
    if ( *(_UNKNOWN **)v12 == &unk_5024E68 && v6 == *(_QWORD *)(v12 + 8) )
      break;
    if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == -4096 )
      goto LABEL_15;
    v13 = v10 + i;
    ++v10;
  }
  if ( v12 == v9 + 24 * v8 )
  {
LABEL_15:
    v14 = 0;
  }
  else
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
    if ( v14 )
    {
      v18 = 1;
      v14 += 8;
      v15 = &v19;
      do
      {
        *v15 = -4096;
        v15 += 2;
      }
      while ( v15 != (__int64 *)&v21 );
      if ( (v18 & 1) == 0 )
        sub_C7D6A0(v19, 16LL * v20, 8);
    }
  }
  v17 = v14;
  sub_2F7C2A0(&v17, a3);
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
