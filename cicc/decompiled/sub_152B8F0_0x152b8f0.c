// Function: sub_152B8F0
// Address: 0x152b8f0
//
void __fastcall sub_152B8F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r12
  int v12; // ecx
  __int64 v13; // r9
  int v14; // ecx
  __int64 v15; // r9
  __int64 v16; // rdi
  unsigned int v17; // esi
  __int64 *v18; // rdx
  __int64 v19; // r11
  int v20; // edx
  int v21; // r10d
  unsigned int v22; // [rsp+4h] [rbp-3Ch]
  __int64 v23; // [rsp+8h] [rbp-38h]

  v8 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v8 )
  {
    v9 = *(unsigned int *)(a3 + 8);
    v10 = (unsigned int)v8;
    v11 = 0;
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 304);
      v13 = 0;
      if ( v12 )
      {
        v14 = v12 - 1;
        v15 = *(_QWORD *)(a1 + 288);
        v16 = *(_QWORD *)(a2 + 8 * (v11 - v8));
        v17 = v14 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v18 = (__int64 *)(v15 + 16LL * v17);
        v19 = *v18;
        if ( v16 == *v18 )
        {
LABEL_6:
          v13 = *((unsigned int *)v18 + 3);
        }
        else
        {
          v20 = 1;
          while ( v19 != -4 )
          {
            v21 = v20 + 1;
            v17 = v14 & (v20 + v17);
            v18 = (__int64 *)(v15 + 16LL * v17);
            v19 = *v18;
            if ( v16 == *v18 )
              goto LABEL_6;
            v20 = v21;
          }
          v13 = 0;
        }
      }
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v9 )
      {
        v22 = a4;
        v23 = v13;
        sub_16CD150(a3, a3 + 16, 0, 8);
        v9 = *(unsigned int *)(a3 + 8);
        a4 = v22;
        v13 = v23;
      }
      ++v11;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v13;
      v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v9;
      if ( v11 == v10 )
        break;
      v8 = *(unsigned int *)(a2 + 8);
    }
  }
  sub_152B6B0(*(_DWORD **)a1, 2 * (*(_BYTE *)(a2 + 1) == 1) + 3, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
