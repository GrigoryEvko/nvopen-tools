// Function: sub_30B8700
// Address: 0x30b8700
//
bool __fastcall sub_30B8700(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r11
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 v20; // rax
  __int64 *v21; // [rsp+0h] [rbp-50h]
  __int64 *v22; // [rsp+0h] [rbp-50h]
  __int64 *v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+0h] [rbp-50h]
  const void *v25; // [rsp+8h] [rbp-48h]
  int v26; // [rsp+10h] [rbp-40h]
  char v27; // [rsp+17h] [rbp-39h]

  v5 = 1;
  v7 = 0;
  v27 = 0;
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v25 = (const void *)(a4 + 16);
  if ( (unsigned int)v8 <= 1 )
    return *(_DWORD *)(a3 + 8) != 0;
  while ( 1 )
  {
    v11 = sub_DD8400(a1, *(_QWORD *)(a2 + 32 * (v5 - v8)));
    v12 = v5;
    if ( v5 != 1 )
      break;
    v7 = *(_QWORD *)(a2 + 72);
    if ( *((_WORD *)v11 + 12) )
      goto LABEL_17;
    v16 = v11[4];
    if ( *(_DWORD *)(v16 + 32) <= 0x40u )
    {
      if ( *(_QWORD *)(v16 + 24) )
      {
LABEL_17:
        v18 = *(unsigned int *)(a3 + 8);
        if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v23 = v11;
          sub_C8D5F0(a3, (const void *)(a3 + 16), v18 + 1, 8u, v9, v10);
          v18 = *(unsigned int *)(a3 + 8);
          v11 = v23;
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v11;
        ++*(_DWORD *)(a3 + 8);
        goto LABEL_9;
      }
    }
    else
    {
      v26 = *(_DWORD *)(v16 + 32);
      v21 = v11;
      v17 = sub_C444A0(v16 + 24);
      v11 = v21;
      if ( v26 != v17 )
        goto LABEL_17;
    }
    v27 = 1;
LABEL_9:
    ++v5;
    v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( (unsigned int)v8 <= (unsigned int)v5 )
      return *(_DWORD *)(a3 + 8) != 0;
  }
  if ( *(_BYTE *)(v7 + 8) == 16 )
  {
    v13 = *(unsigned int *)(a3 + 8);
    v14 = v13 + 1;
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v22 = v11;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v13 + 1, 8u, a3 + 16, v14);
      v13 = *(unsigned int *)(a3 + 8);
      v12 = v5;
      v11 = v22;
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = v11;
    ++*(_DWORD *)(a3 + 8);
    if ( !v27 || v12 != 2 )
    {
      v19 = *(_QWORD *)(v7 + 32);
      v20 = *(unsigned int *)(a4 + 8);
      if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        v24 = *(_QWORD *)(v7 + 32);
        sub_C8D5F0(a4, v25, v20 + 1, 4u, v9, v14);
        LODWORD(v19) = v24;
        v20 = *(unsigned int *)(a4 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a4 + 4 * v20) = v19;
      ++*(_DWORD *)(a4 + 8);
    }
    v7 = *(_QWORD *)(v7 + 24);
    goto LABEL_9;
  }
  *(_DWORD *)(a3 + 8) = 0;
  *(_DWORD *)(a4 + 8) = 0;
  return 0;
}
