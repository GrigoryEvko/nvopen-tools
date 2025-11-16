// Function: sub_152D3D0
// Address: 0x152d3d0
//
void __fastcall sub_152D3D0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rax
  _BOOL8 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *i; // r12
  int v12; // ecx
  __int64 v13; // r8
  int v14; // ecx
  __int64 v15; // r8
  unsigned int v16; // esi
  __int64 *v17; // rdx
  __int64 v18; // r10
  int v19; // edx
  int v20; // r9d
  __int64 v21; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned int *)(a3 + 8);
  v8 = *(_BYTE *)(a2 + 1) == 1;
  if ( (unsigned int)v7 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v8;
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = 8LL * *(unsigned int *)(a2 + 8);
  for ( i = (_QWORD *)(a2 - v10); (_QWORD *)a2 != i; *(_DWORD *)(a3 + 8) = v9 )
  {
    v12 = *(_DWORD *)(a1 + 304);
    v13 = 0;
    if ( v12 )
    {
      v14 = v12 - 1;
      v15 = *(_QWORD *)(a1 + 288);
      v16 = v14 & (((unsigned int)*i >> 9) ^ ((unsigned int)*i >> 4));
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( *i == *v17 )
      {
LABEL_6:
        v13 = *((unsigned int *)v17 + 3);
      }
      else
      {
        v19 = 1;
        while ( v18 != -4 )
        {
          v20 = v19 + 1;
          v16 = v14 & (v19 + v16);
          v17 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( *i == *v17 )
            goto LABEL_6;
          v19 = v20;
        }
        v13 = 0;
      }
    }
    if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v9 )
    {
      v21 = v13;
      sub_16CD150(a3, a3 + 16, 0, 8);
      v9 = *(unsigned int *)(a3 + 8);
      v13 = v21;
    }
    ++i;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v13;
    v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  }
  sub_152B6B0(*(_DWORD **)a1, 0x20u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
