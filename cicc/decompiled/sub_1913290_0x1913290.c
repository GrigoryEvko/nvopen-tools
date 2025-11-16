// Function: sub_1913290
// Address: 0x1913290
//
__int64 __fastcall sub_1913290(__int64 a1, __int64 a2, __int64 *a3)
{
  const void *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  int v7; // r8d
  int v8; // r9d
  int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rax
  int *v12; // r14
  int *i; // r15
  int v14; // ecx
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rax
  int v20; // ebx
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // rax
  int v24; // [rsp+Ch] [rbp-34h]

  v4 = (const void *)(a1 + 40);
  *(_DWORD *)a1 = -3;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  v5 = *a3;
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v5;
  v6 = *(a3 - 3);
  if ( *(_BYTE *)(v6 + 16) == 78
    && (v16 = *(_QWORD *)(v6 - 24), !*(_BYTE *)(v16 + 16))
    && (*(_BYTE *)(v16 + 33) & 0x20) != 0
    && *((_DWORD *)a3 + 16) == 1
    && !*(_DWORD *)a3[7] )
  {
    switch ( *(_DWORD *)(v16 + 36) )
    {
      case 0xBD:
      case 0xD1:
        *(_DWORD *)a1 = 11;
        break;
      case 0xC3:
      case 0xD2:
        *(_DWORD *)a1 = 15;
        break;
      case 0xC6:
      case 0xD3:
        *(_DWORD *)a1 = 13;
        break;
      default:
        goto LABEL_2;
    }
    v17 = sub_1911FD0(a2, *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
    v19 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v19 >= *(_DWORD *)(a1 + 36) )
    {
      v24 = v17;
      sub_16CD150(a1 + 24, v4, 0, 4, v17, v18);
      v19 = *(unsigned int *)(a1 + 32);
      v17 = v24;
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v19) = v17;
    ++*(_DWORD *)(a1 + 32);
    v20 = sub_1911FD0(a2, *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF))));
    v23 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v23 >= *(_DWORD *)(a1 + 36) )
    {
      sub_16CD150(a1 + 24, v4, 0, 4, v21, v22);
      v23 = *(unsigned int *)(a1 + 32);
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v23) = v20;
    ++*(_DWORD *)(a1 + 32);
  }
  else
  {
LABEL_2:
    *(_DWORD *)a1 = *((unsigned __int8 *)a3 + 16) - 24;
    v9 = sub_1911FD0(a2, *(a3 - 3));
    v10 = *(unsigned int *)(a1 + 32);
    if ( *(_DWORD *)(a1 + 36) <= (unsigned int)v10 )
    {
      sub_16CD150(a1 + 24, v4, 0, 4, v7, v8);
      v10 = *(unsigned int *)(a1 + 32);
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v10) = v9;
    v11 = (unsigned int)(*(_DWORD *)(a1 + 32) + 1);
    *(_DWORD *)(a1 + 32) = v11;
    v12 = (int *)a3[7];
    for ( i = &v12[*((unsigned int *)a3 + 16)]; v12 != i; *(_DWORD *)(a1 + 32) = v11 )
    {
      if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 36) )
      {
        sub_16CD150(a1 + 24, v4, 0, 4, v7, v8);
        v11 = *(unsigned int *)(a1 + 32);
      }
      v14 = *v12++;
      *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v11) = v14;
      v11 = (unsigned int)(*(_DWORD *)(a1 + 32) + 1);
    }
  }
  return a1;
}
