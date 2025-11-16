// Function: sub_2794270
// Address: 0x2794270
//
__int64 __fastcall sub_2794270(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 *v11; // r14
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rbx
  const void *v17; // r13
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // r8
  __int64 v22; // r9
  int v23; // r12d
  __int64 v24; // rax
  int v25; // ebx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  int v29; // [rsp+Ch] [rbp-44h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  const void *v31; // [rsp+18h] [rbp-38h]

  *(_QWORD *)(a1 + 16) = a1 + 32;
  v31 = (const void *)(a1 + 32);
  *(_DWORD *)a1 = -3;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_QWORD *)(a1 + 48) = 0;
  v6 = *((_QWORD *)a3 + 1);
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v6;
  v7 = *((_QWORD *)a3 - 4);
  v30 = a1 + 16;
  if ( *(_BYTE *)v7 == 85 )
  {
    v19 = *(_QWORD *)(v7 - 32);
    if ( v19 )
    {
      if ( !*(_BYTE *)v19 && *(_QWORD *)(v19 + 24) == *(_QWORD *)(v7 + 80) && (*(_BYTE *)(v19 + 33) & 0x20) != 0 )
      {
        v20 = *(_DWORD *)(v19 + 36);
        if ( v20 != 312 )
        {
          switch ( v20 )
          {
            case 333:
            case 339:
            case 360:
            case 369:
            case 372:
              break;
            default:
              goto LABEL_2;
          }
        }
        if ( *((_DWORD *)a3 + 20) == 1 && !**((_DWORD **)a3 + 9) )
        {
          *(_DWORD *)a1 = sub_B5B5E0(v7);
          v23 = sub_2792F80(a2, *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
          v24 = *(unsigned int *)(a1 + 24);
          if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
          {
            sub_C8D5F0(v30, v31, v24 + 1, 4u, v21, v22);
            v24 = *(unsigned int *)(a1 + 24);
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v24) = v23;
          ++*(_DWORD *)(a1 + 24);
          v25 = sub_2792F80(a2, *(_QWORD *)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))));
          v28 = *(unsigned int *)(a1 + 24);
          if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
          {
            sub_C8D5F0(v30, v31, v28 + 1, 4u, v26, v27);
            v28 = *(unsigned int *)(a1 + 24);
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v28) = v25;
          ++*(_DWORD *)(a1 + 24);
          return a1;
        }
      }
    }
  }
LABEL_2:
  *(_DWORD *)a1 = *a3 - 29;
  if ( (a3[7] & 0x40) != 0 )
  {
    v10 = *((_QWORD *)a3 - 1);
    v11 = (__int64 *)v10;
    v8 = (__int64 *)(v10 + 32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF));
    if ( (__int64 *)v10 != v8 )
      goto LABEL_4;
  }
  else
  {
    v8 = (__int64 *)a3;
    v9 = 32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF);
    v10 = (__int64)&a3[-v9];
    v11 = (__int64 *)&a3[-v9];
    if ( &a3[-v9] != a3 )
    {
      do
      {
LABEL_4:
        v12 = sub_2792F80(a2, *v11);
        v13 = *(unsigned int *)(a1 + 24);
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
        {
          v29 = v12;
          sub_C8D5F0(v30, v31, v13 + 1, 4u, a5, v10);
          v13 = *(unsigned int *)(a1 + 24);
          v12 = v29;
        }
        v11 += 4;
        *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v13) = v12;
        v14 = (unsigned int)(*(_DWORD *)(a1 + 24) + 1);
        *(_DWORD *)(a1 + 24) = v14;
      }
      while ( v8 != v11 );
      v15 = *(unsigned int *)(a1 + 28);
      goto LABEL_8;
    }
  }
  v15 = 4;
  v14 = 0;
LABEL_8:
  v16 = *((unsigned int *)a3 + 20);
  v17 = (const void *)*((_QWORD *)a3 + 9);
  if ( v16 + v14 > v15 )
  {
    sub_C8D5F0(v30, v31, v16 + v14, 4u, a5, v10);
    v14 = *(unsigned int *)(a1 + 24);
  }
  if ( 4 * v16 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 16) + 4 * v14), v17, 4 * v16);
    LODWORD(v14) = *(_DWORD *)(a1 + 24);
  }
  *(_DWORD *)(a1 + 24) = v16 + v14;
  return a1;
}
