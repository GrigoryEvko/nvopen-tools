// Function: sub_1376790
// Address: 0x1376790
//
void __fastcall sub_1376790(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  unsigned __int64 v5; // rdx
  int *v6; // r15
  int *v7; // rbx
  __int64 v8; // rax
  _DWORD *v9; // rdi
  __int64 v10; // r9
  __int64 v11; // rax
  _QWORD *v12; // r13
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 *v16; // r12
  __int64 v17; // rsi
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // r15
  __int64 *v21; // rdi
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 *v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  _DWORD v26[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v4 = *(_QWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 8) = **(_DWORD **)(a2 + 96);
  v5 = *(unsigned int *)(a2 + 104);
  v23 = (__int64 *)(a1 + 24);
  if ( v5 > 0x2E8BA2E8BA2E8BA3LL * ((*(_QWORD *)(a1 + 40) - v4) >> 3) )
  {
    v17 = *(_QWORD *)(a1 + 32);
    v18 = 0;
    v25 = v17 - v4;
    v22 = 88 * v5;
    if ( v5 )
      v18 = sub_22077B0(88 * v5);
    sub_13761C0(v4, v17, v18);
    v19 = *(_QWORD *)(a1 + 32);
    v20 = *(_QWORD *)(a1 + 24);
    if ( v19 != v20 )
    {
      do
      {
        v21 = (__int64 *)(v20 + 8);
        v20 += 88;
        sub_13713C0(v21);
      }
      while ( v19 != v20 );
      v20 = *(_QWORD *)(a1 + 24);
    }
    if ( v20 )
      j_j___libc_free_0(v20, *(_QWORD *)(a1 + 40) - v20);
    *(_QWORD *)(a1 + 24) = v18;
    *(_QWORD *)(a1 + 32) = v18 + v25;
    *(_QWORD *)(a1 + 40) = v22 + v18;
    v5 = *(unsigned int *)(a2 + 104);
  }
  v6 = *(int **)(a2 + 96);
  v7 = &v6[v5];
  while ( v7 != v6 )
  {
    v13 = *v6;
    v14 = *(_QWORD *)(a1 + 32);
    v26[0] = *v6;
    if ( v14 == *(_QWORD *)(a1 + 40) )
    {
      sub_13763B0(v23, v14, v26);
    }
    else
    {
      if ( v14 )
      {
        *(_DWORD *)v14 = v13;
        *(_DWORD *)(v14 + 4) = 0;
        *(_QWORD *)(v14 + 8) = 0;
        *(_QWORD *)(v14 + 16) = 0;
        *(_QWORD *)(v14 + 24) = 0;
        *(_QWORD *)(v14 + 32) = 0;
        *(_QWORD *)(v14 + 40) = 0;
        *(_QWORD *)(v14 + 48) = 0;
        *(_QWORD *)(v14 + 56) = 0;
        *(_QWORD *)(v14 + 64) = 0;
        *(_QWORD *)(v14 + 72) = 0;
        *(_QWORD *)(v14 + 80) = 0;
        sub_1371810((__int64 *)(v14 + 8), 0);
        v14 = *(_QWORD *)(a1 + 32);
      }
      *(_QWORD *)(a1 + 32) = v14 + 88;
    }
    v15 = *(_QWORD *)(*(_QWORD *)a1 + 64LL) + 24LL * v26[0];
    v16 = *(__int64 **)(v15 + 8);
    if ( v16 )
    {
      v8 = *((unsigned int *)v16 + 3);
      v9 = (_DWORD *)v16[12];
      if ( (unsigned int)v8 > 1 )
      {
        if ( !sub_1369030(v9, &v9[v8], (_DWORD *)v15) )
        {
          v12 = (_QWORD *)(v15 + 16);
          goto LABEL_10;
        }
LABEL_6:
        if ( *((_BYTE *)v16 + 8) )
        {
          v10 = *v16;
          if ( !*v16
            || (v11 = *(unsigned int *)(v10 + 12), (unsigned int)v11 <= 1)
            || (v24 = *v16,
                !sub_1369030(*(_DWORD **)(v10 + 96), (_DWORD *)(*(_QWORD *)(v10 + 96) + 4 * v11), (_DWORD *)v15))
            || (v12 = (_QWORD *)(v24 + 152), !*(_BYTE *)(v24 + 8)) )
          {
            v12 = v16 + 19;
          }
          goto LABEL_10;
        }
        goto LABEL_16;
      }
      if ( *(_DWORD *)v15 == *v9 )
        goto LABEL_6;
    }
LABEL_16:
    v12 = (_QWORD *)(v15 + 16);
LABEL_10:
    ++v6;
    *v12 = 0;
  }
  sub_1373CE0(a1);
}
