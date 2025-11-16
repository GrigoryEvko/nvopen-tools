// Function: sub_3434150
// Address: 0x3434150
//
_QWORD *__fastcall sub_3434150(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // eax
  unsigned __int64 v11; // r8
  int v12; // r10d
  char v13; // di
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // r9
  unsigned int v16; // eax
  __int64 v17; // r14
  unsigned __int64 v18; // r11
  __int64 v19; // rdx
  int v20; // r9d
  _QWORD *v21; // rax
  __int64 v22; // r9
  __int64 v23; // r8
  _QWORD *v24; // r14
  __int64 v25; // r12
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // rsi
  unsigned int v30; // [rsp+Ch] [rbp-64h]
  __int64 v31; // [rsp+10h] [rbp-60h] BYREF
  __int64 v32; // [rsp+18h] [rbp-58h]
  unsigned __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  char v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h]
  __int64 v36; // [rsp+38h] [rbp-38h]

  v32 = a3;
  v6 = *(_QWORD *)(a4 + 864);
  v31 = a2;
  v7 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 48LL);
  if ( (_WORD)a2 )
  {
    if ( (_WORD)a2 == 1 || (unsigned __int16)(a2 - 504) <= 7u )
      BUG();
    v8 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a2 - 16];
    LOBYTE(v9) = byte_444C4A0[16 * (unsigned __int16)a2 - 8];
  }
  else
  {
    v8 = sub_3007260((__int64)&v31);
    v35 = v8;
    v36 = v9;
  }
  v34 = v9;
  v33 = (unsigned __int64)(v8 + 7) >> 3;
  v10 = sub_CA1930(&v33);
  v11 = *(_QWORD *)(a1 + 32);
  v12 = v10;
  v13 = v11 & 1;
  if ( (*(_QWORD *)(a1 + 32) & 1) != 0 )
    v14 = v11 >> 58;
  else
    v14 = *(unsigned int *)(v11 + 64);
  v15 = *(unsigned int *)(a1 + 40);
  v16 = *(_DWORD *)(a1 + 40);
  if ( v14 <= v15 )
  {
LABEL_13:
    v21 = sub_33EDFE0(*(_QWORD *)(a4 + 864), (unsigned int)v31, v32, 1, v11, v15);
    v23 = *((unsigned int *)v21 + 24);
    v24 = v21;
    *(_BYTE *)(*(_QWORD *)(v7 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v7 + 32) + *((_DWORD *)v21 + 24)) + 19) = 1;
    v25 = *(_QWORD *)(a4 + 960);
    v26 = *(unsigned int *)(v25 + 536);
    v27 = *(unsigned int *)(v25 + 540);
    if ( v26 + 1 > v27 )
    {
      v30 = v23;
      sub_C8D5F0(v25 + 528, (const void *)(v25 + 544), v26 + 1, 4u, v23, v22);
      v26 = *(unsigned int *)(v25 + 536);
      v23 = v30;
    }
    *(_DWORD *)(*(_QWORD *)(v25 + 528) + 4 * v26) = v23;
    ++*(_DWORD *)(v25 + 536);
    v28 = *(_QWORD *)(a1 + 32);
    if ( (v28 & 1) != 0 )
      v28 >>= 58;
    else
      LODWORD(v28) = *(_DWORD *)(v28 + 64);
    sub_228BF90((unsigned __int64 *)(a1 + 32), v28 + 1, 1u, v27, v23, v22);
    return v24;
  }
  else
  {
    v17 = ~(-1LL << (v11 >> 58));
    v18 = v17 & (v11 >> 1);
    while ( 1 )
    {
      v19 = v13 ? (v18 >> v16) & 1 : (*(_QWORD *)(*(_QWORD *)v11 + 8LL * (v16 >> 6)) >> v16) & 1LL;
      if ( !(_BYTE)v19 )
      {
        v20 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 960) + 528LL) + 4 * v15);
        if ( *(_QWORD *)(*(_QWORD *)(v7 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v7 + 32) + v20) + 8) == v12 )
          break;
      }
      v15 = v16 + 1;
      *(_DWORD *)(a1 + 40) = v15;
      ++v16;
      if ( v15 >= v14 )
        goto LABEL_13;
    }
    if ( v13 )
      *(_QWORD *)(a1 + 32) = 2 * ((v18 | (1LL << v16)) & v17 | (v11 >> 58 << 57)) + 1;
    else
      *(_QWORD *)(*(_QWORD *)v11 + 8LL * (v16 >> 6)) |= 1LL << v16;
    return sub_33EDBD0(*(_QWORD **)(a4 + 864), v20, v31, v32, 0);
  }
}
