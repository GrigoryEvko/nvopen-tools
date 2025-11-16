// Function: sub_2D32F70
// Address: 0x2d32f70
//
__int64 __fastcall sub_2D32F70(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // rcx
  unsigned int v14; // eax
  __int64 *v15; // r15
  int v16; // eax
  unsigned int v17; // esi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 *v24; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v25; // [rsp+8h] [rbp-38h] BYREF

  if ( (unsigned __int8)sub_2D29100(a2, a3, &v24) )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v9 = a2 + 16;
      v10 = 352;
    }
    else
    {
      v9 = *(_QWORD *)(a2 + 16);
      v10 = 88LL * *(unsigned int *)(a2 + 24);
    }
    v11 = *(_QWORD *)a2;
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 24) = v10 + v9;
    *(_QWORD *)(a1 + 8) = v11;
    v12 = v24;
    *(_BYTE *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 16) = v12;
    return a1;
  }
  v14 = *(_DWORD *)(a2 + 8);
  v15 = v24;
  ++*(_QWORD *)a2;
  v25 = v15;
  v16 = (v14 >> 1) + 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v18 = 12;
    v17 = 4;
  }
  else
  {
    v17 = *(_DWORD *)(a2 + 24);
    v18 = 3 * v17;
  }
  v19 = (unsigned int)(4 * v16);
  if ( (unsigned int)v18 <= (unsigned int)v19 )
  {
    v17 *= 2;
  }
  else
  {
    v18 = v17 - (v16 + *(_DWORD *)(a2 + 12));
    v19 = v17 >> 3;
    if ( (unsigned int)v18 > (unsigned int)v19 )
      goto LABEL_11;
  }
  sub_2D32AA0(a2, v17, v19, v18, v7, v8);
  sub_2D29100(a2, a3, &v25);
  v15 = v25;
  v16 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
LABEL_11:
  v20 = *(_DWORD *)(a2 + 8) & 1;
  *(_DWORD *)(a2 + 8) = v20 | (2 * v16);
  if ( *v15 != -4096 || v15[1] != -4096 )
    --*(_DWORD *)(a2 + 12);
  *v15 = *a3;
  v15[1] = a3[1];
  v15[2] = (__int64)(v15 + 4);
  v15[3] = 0x600000000LL;
  if ( *(_DWORD *)(a4 + 8) )
    sub_2D23900((__int64)(v15 + 2), (char **)a4, v20, v18, v7, v8);
  *((_DWORD *)v15 + 20) = *(_DWORD *)(a4 + 64);
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v21 = a2 + 16;
    v22 = 352;
  }
  else
  {
    v21 = *(_QWORD *)(a2 + 16);
    v22 = 88LL * *(unsigned int *)(a2 + 24);
  }
  v23 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v15;
  *(_QWORD *)(a1 + 8) = v23;
  *(_QWORD *)(a1 + 24) = v22 + v21;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
