// Function: sub_2B1F480
// Address: 0x2b1f480
//
__int64 __fastcall sub_2B1F480(__int64 a1, unsigned __int8 *a2, unsigned int a3, int a4, unsigned int a5, __int64 a6)
{
  int v8; // edx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // ebx
  __int64 v17; // rax
  unsigned __int8 *v18; // r14
  int v19; // r13d
  int v20; // esi
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // eax
  __int64 v30; // r10
  __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // eax
  const void *v35; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v41; // [rsp+28h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  v35 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  v8 = *a2;
  if ( v8 == 40 )
  {
    v9 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v9 = -32;
    if ( v8 != 85 )
    {
      v9 = -96;
      if ( v8 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v10 = sub_BD2BC0((__int64)a2);
    v12 = v10 + v11;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v12 >> 4) )
        goto LABEL_9;
    }
    else
    {
      if ( !(unsigned int)((v12 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_9;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v13 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v14 = sub_BD2BC0((__int64)a2);
        v9 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
        goto LABEL_9;
      }
    }
    BUG();
  }
LABEL_9:
  v16 = 0;
  v41 = &a2[v9];
  v17 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( &a2[-v17] != &a2[v9] )
  {
    v18 = &a2[-v17];
    v19 = a4;
    do
    {
      if ( !a3 )
        goto LABEL_20;
      if ( sub_9B75A0(a3, v16, a6) )
      {
        v30 = *(_QWORD *)(*(_QWORD *)v18 + 8LL);
        v31 = *(unsigned int *)(a1 + 8);
        if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v38 = *(_QWORD *)(*(_QWORD *)v18 + 8LL);
          sub_C8D5F0(a1, v35, v31 + 1, 8u, v26, v27);
          v31 = *(unsigned int *)(a1 + 8);
          v30 = v38;
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v31) = v30;
        ++*(_DWORD *)(a1 + 8);
        goto LABEL_16;
      }
      if ( a5 )
      {
        v32 = (_QWORD *)sub_BD5C60((__int64)a2);
        v21 = sub_BCCE00(v32, a5);
        v33 = *(unsigned __int8 *)(v21 + 8);
        if ( (_BYTE)v33 != 17 )
        {
          v20 = v19;
          if ( (unsigned int)(v33 - 17) > 1 )
            goto LABEL_13;
          goto LABEL_12;
        }
      }
      else
      {
LABEL_20:
        v21 = *(_QWORD *)(*(_QWORD *)v18 + 8LL);
        v28 = *(unsigned __int8 *)(v21 + 8);
        if ( (_BYTE)v28 != 17 )
        {
          v20 = v19;
          if ( (unsigned int)(v28 - 17) > 1 )
            goto LABEL_13;
          goto LABEL_12;
        }
      }
      v20 = v19 * *(_DWORD *)(v21 + 32);
LABEL_12:
      v21 = **(_QWORD **)(v21 + 16);
LABEL_13:
      v22 = sub_BCDA70((__int64 *)v21, v20);
      v25 = *(unsigned int *)(a1 + 8);
      if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v37 = v22;
        sub_C8D5F0(a1, v35, v25 + 1, 8u, v23, v24);
        v25 = *(unsigned int *)(a1 + 8);
        v22 = v37;
      }
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v25) = v22;
      ++*(_DWORD *)(a1 + 8);
LABEL_16:
      ++v16;
      v18 += 32;
    }
    while ( v41 != v18 );
  }
  return a1;
}
