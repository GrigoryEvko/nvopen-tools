// Function: sub_27EE2B0
// Address: 0x27ee2b0
//
void __fastcall sub_27EE2B0(__int64 a1, int a2, unsigned int a3, char a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rdi
  __int64 *v8; // rbx
  __int64 v9; // r11
  int v11; // eax
  __int64 v12; // r9
  int v13; // r12d
  __int64 v14; // r13
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rcx
  __int64 v19; // rdx
  int v20; // edx
  int v21; // r14d

  *(_BYTE *)a1 = 0;
  *(_DWORD *)(a1 + 4) = 0;
  *(_DWORD *)(a1 + 8) = a2;
  *(_DWORD *)(a1 + 12) = a3;
  *(_BYTE *)(a1 + 16) = a4;
  v7 = *(__int64 **)(a5 + 32);
  v8 = *(__int64 **)(a5 + 40);
  if ( v7 == v8 )
    return;
  v9 = *(_QWORD *)(a6 + 72);
  v11 = 0;
  v12 = *(unsigned int *)(a6 + 88);
  v13 = v12 - 1;
  while ( 1 )
  {
    v14 = *v7;
    if ( (_DWORD)v12 )
    {
      v15 = v13 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v16 = (__int64 *)(v9 + 16LL * v15);
      v17 = *v16;
      if ( v14 != *v16 )
      {
        v20 = 1;
        while ( v17 != -4096 )
        {
          v21 = v20 + 1;
          v15 = v13 & (v20 + v15);
          v16 = (__int64 *)(v9 + 16LL * v15);
          v17 = *v16;
          if ( v14 == *v16 )
            goto LABEL_5;
          v20 = v21;
        }
        goto LABEL_15;
      }
LABEL_5:
      if ( (__int64 *)(v9 + 16 * v12) != v16 )
      {
        v18 = v16[1];
        if ( v18 )
        {
          v19 = *(_QWORD *)(v18 + 8);
          if ( v18 != v19 )
            break;
        }
      }
    }
LABEL_15:
    if ( v8 == ++v7 )
      return;
  }
  while ( a3 >= ++v11 )
  {
    v19 = *(_QWORD *)(v19 + 8);
    if ( v18 == v19 )
      goto LABEL_15;
  }
  *(_BYTE *)a1 = 1;
}
