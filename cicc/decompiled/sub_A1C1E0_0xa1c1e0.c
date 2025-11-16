// Function: sub_A1C1E0
// Address: 0xa1c1e0
//
void __fastcall sub_A1C1E0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int8 v8; // al
  __int64 *v9; // r12
  __int64 v10; // rdx
  __int64 *v11; // r14
  __int64 v12; // rax
  int v13; // esi
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r11
  __int64 v17; // r9
  __int64 v18; // rdx
  int v19; // esi
  __int64 v20; // rdi
  __int64 v21; // r9
  int v22; // edx
  int v23; // r10d
  unsigned int v24; // [rsp+4h] [rbp-3Ch]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
  {
    v9 = *(__int64 **)(a2 - 32);
    v10 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    v10 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    v9 = (__int64 *)(a2 - 16 - 8LL * ((v8 >> 2) & 0xF));
  }
  v11 = &v9[v10];
  if ( v9 != v11 )
  {
    v12 = *(unsigned int *)(a3 + 8);
    do
    {
      v19 = *(_DWORD *)(a1 + 304);
      v20 = *v9;
      v21 = *(_QWORD *)(a1 + 288);
      if ( v19 )
      {
        v13 = v19 - 1;
        v14 = v13 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v15 = (__int64 *)(v21 + 16LL * v14);
        v16 = *v15;
        if ( v20 == *v15 )
        {
LABEL_6:
          v17 = *((unsigned int *)v15 + 3);
          v18 = v12 + 1;
          if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
            goto LABEL_10;
          goto LABEL_7;
        }
        v22 = 1;
        while ( v16 != -4096 )
        {
          v23 = v22 + 1;
          v14 = v13 & (v22 + v14);
          v15 = (__int64 *)(v21 + 16LL * v14);
          v16 = *v15;
          if ( v20 == *v15 )
            goto LABEL_6;
          v22 = v23;
        }
      }
      v18 = v12 + 1;
      v17 = 0;
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
LABEL_10:
        v24 = a4;
        v25 = v17;
        sub_C8D5F0(a3, a3 + 16, v18, 8);
        v12 = *(unsigned int *)(a3 + 8);
        a4 = v24;
        v17 = v25;
      }
LABEL_7:
      ++v9;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v17;
      v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v12;
    }
    while ( v11 != v9 );
  }
  sub_A1BFB0(*(_QWORD *)a1, 2 * ((*(_BYTE *)(a2 + 1) & 0x7F) == 1) + 3, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
