// Function: sub_1390580
// Address: 0x1390580
//
__int64 __fastcall sub_1390580(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax
  _QWORD *v11; // r14
  __int64 v12; // rax
  _QWORD *v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rsi
  int v16; // r10d
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned int i; // eax
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rbx
  unsigned int j; // r14d
  __int64 v26; // rdx
  __int64 v27; // rcx
  int v28; // r9d
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rsi
  unsigned int k; // eax
  __int64 v32; // rsi
  unsigned int v33; // eax
  __int64 v34[2]; // [rsp-58h] [rbp-58h] BYREF
  __int64 v35; // [rsp-48h] [rbp-48h] BYREF
  __int64 v36; // [rsp-40h] [rbp-40h]
  __int64 v37; // [rsp-38h] [rbp-38h]
  int v38; // [rsp-30h] [rbp-30h]

  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v4 = *(_QWORD *)(a4 + 8);
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 8) = v4;
  LODWORD(v4) = *(_DWORD *)(a4 + 16);
  ++*(_QWORD *)a4;
  *(_DWORD *)(a1 + 16) = v4;
  LODWORD(v4) = *(_DWORD *)(a4 + 20);
  *(_QWORD *)(a4 + 8) = 0;
  *(_DWORD *)(a1 + 20) = v4;
  LODWORD(v4) = *(_DWORD *)(a4 + 24);
  *(_QWORD *)(a4 + 16) = 0;
  *(_DWORD *)(a1 + 24) = v4;
  v5 = *(_QWORD *)(a4 + 32);
  *(_DWORD *)(a4 + 24) = 0;
  *(_QWORD *)(a1 + 32) = v5;
  v6 = *(_QWORD *)(a4 + 40);
  *(_QWORD *)(a4 + 32) = 0;
  *(_QWORD *)(a1 + 40) = v6;
  v7 = *(_QWORD *)(a4 + 48);
  *(_QWORD *)(a4 + 40) = 0;
  *(_QWORD *)(a1 + 48) = v7;
  *(_QWORD *)(a4 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 72;
  result = 0x800000000LL;
  *(_QWORD *)(a1 + 64) = 0x800000000LL;
  *(_QWORD *)(a1 + 264) = a1 + 280;
  *(_QWORD *)(a1 + 272) = 0x800000000LL;
  if ( *(_QWORD *)(a2 + 96) <= 0x32u )
  {
    v34[0] = (__int64)&v35;
    v11 = *(_QWORD **)a3;
    v12 = *(unsigned int *)(a3 + 8);
    v35 = 0;
    v36 = 0;
    v13 = &v11[v12];
    v37 = 0;
    v38 = 0;
    for ( v34[1] = a1; v13 != v11; ++v11 )
    {
      v14 = *(unsigned int *)(a1 + 24);
      if ( (_DWORD)v14 )
      {
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = ((((unsigned __int64)(((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4)) << 32) - 1) >> 22)
            ^ (((unsigned __int64)(((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4)) << 32) - 1);
        v18 = ((9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13)))) >> 15)
            ^ (9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13))));
        for ( i = (v14 - 1) & (((v18 - 1 - (v18 << 27)) >> 31) ^ (v18 - 1 - ((_DWORD)v18 << 27))); ; i = (v14 - 1) & v21 )
        {
          v20 = v15 + 24LL * i;
          if ( *v11 == *(_QWORD *)v20 && !*(_DWORD *)(v20 + 8) )
            break;
          if ( *(_QWORD *)v20 == -8 && *(_DWORD *)(v20 + 8) == -1 )
            goto LABEL_11;
          v21 = v16 + i;
          ++v16;
        }
        if ( v20 != v15 + 24 * v14 )
          sub_1390160(v34, 0, *(_DWORD *)(v20 + 16));
      }
LABEL_11:
      ;
    }
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2);
      v22 = *(_QWORD *)(a2 + 88);
      v23 = v22 + 40LL * *(_QWORD *)(a2 + 96);
      if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      {
        sub_15E08E0(a2);
        v22 = *(_QWORD *)(a2 + 88);
      }
    }
    else
    {
      v22 = *(_QWORD *)(a2 + 88);
      v23 = v22 + 40LL * *(_QWORD *)(a2 + 96);
    }
    v24 = v22;
    for ( j = 0; v23 != v24; v24 += 40 )
    {
      ++j;
      if ( *(_BYTE *)(*(_QWORD *)v24 + 8LL) == 15 )
      {
        v26 = *(unsigned int *)(a1 + 24);
        if ( (_DWORD)v26 )
        {
          v27 = *(_QWORD *)(a1 + 8);
          v28 = 1;
          v29 = ((((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32) - 1) >> 22)
              ^ (((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32) - 1);
          v30 = ((9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13)))) >> 15)
              ^ (9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13))));
          for ( k = (v26 - 1) & (((v30 - 1 - (v30 << 27)) >> 31) ^ (v30 - 1 - ((_DWORD)v30 << 27))); ; k = (v26 - 1) & v33 )
          {
            v32 = v27 + 24LL * k;
            if ( *(_QWORD *)v32 == v24 && !*(_DWORD *)(v32 + 8) )
              break;
            if ( *(_QWORD *)v32 == -8 && *(_DWORD *)(v32 + 8) == -1 )
              goto LABEL_16;
            v33 = v28 + k;
            ++v28;
          }
          if ( v32 != v27 + 24 * v26 )
            sub_1390160(v34, j, *(_DWORD *)(v32 + 16));
        }
      }
LABEL_16:
      ;
    }
    return j___libc_free_0(v36);
  }
  return result;
}
