// Function: sub_22AE820
// Address: 0x22ae820
//
__int64 __fastcall sub_22AE820(__int64 a1, __int64 a2)
{
  unsigned int v4; // ecx
  __int64 v5; // rsi
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  int v10; // r14d
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 *v13; // r15
  __int64 *v14; // r12
  __int64 v15; // r8
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r10
  __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // r9
  int v23; // edx
  int v24; // eax
  int v25; // r10d
  int v26; // [rsp+4h] [rbp-3Ch]
  __int64 v27; // [rsp+8h] [rbp-38h]
  int v28; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL);
  if ( v4 )
  {
    v7 = (v4 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( v6 == *v8 )
      goto LABEL_3;
    v24 = 1;
    while ( v9 != -4096 )
    {
      v25 = v24 + 1;
      v7 = (v4 - 1) & (v24 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( v6 == *v8 )
        goto LABEL_3;
      v24 = v25;
    }
  }
  v8 = (__int64 *)(v5 + 16LL * v4);
LABEL_3:
  v10 = *((_DWORD *)v8 + 2);
  result = sub_22AE7B0(a1);
  v13 = (__int64 *)(result + 8 * v12);
  v14 = (__int64 *)result;
  if ( (__int64 *)result != v13 )
  {
    result = *(unsigned int *)(a1 + 136);
    v15 = a1 + 128;
    while ( 1 )
    {
      v20 = *(unsigned int *)(a2 + 24);
      v21 = *v14;
      v22 = *(_QWORD *)(a2 + 8);
      if ( !(_DWORD)v20 )
        goto LABEL_10;
      v16 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v17 = (__int64 *)(v22 + 16LL * v16);
      v18 = *v17;
      if ( v21 != *v17 )
        break;
LABEL_6:
      v19 = (unsigned int)(*((_DWORD *)v17 + 2) - v10);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
      {
        v26 = *((_DWORD *)v17 + 2) - v10;
        v27 = v15;
        sub_C8D5F0(v15, (const void *)(a1 + 144), result + 1, 4u, v15, v19);
        result = *(unsigned int *)(a1 + 136);
        LODWORD(v19) = v26;
        v15 = v27;
      }
      ++v14;
      *(_DWORD *)(*(_QWORD *)(a1 + 128) + 4 * result) = v19;
      result = (unsigned int)(*(_DWORD *)(a1 + 136) + 1);
      *(_DWORD *)(a1 + 136) = result;
      if ( v13 == v14 )
        return result;
    }
    v23 = 1;
    while ( v18 != -4096 )
    {
      v16 = (v20 - 1) & (v23 + v16);
      v28 = v23 + 1;
      v17 = (__int64 *)(v22 + 16LL * v16);
      v18 = *v17;
      if ( v21 == *v17 )
        goto LABEL_6;
      v23 = v28;
    }
LABEL_10:
    v17 = (__int64 *)(v22 + 16 * v20);
    goto LABEL_6;
  }
  return result;
}
