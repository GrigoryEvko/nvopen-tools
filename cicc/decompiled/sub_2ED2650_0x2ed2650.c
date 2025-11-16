// Function: sub_2ED2650
// Address: 0x2ed2650
//
__int64 __fastcall sub_2ED2650(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, _QWORD *a5)
{
  __int64 v6; // r10
  __int64 v9; // rbx
  int v11; // edi
  __int16 *v12; // rax
  unsigned int v13; // ecx
  unsigned int v14; // ecx
  __int16 *v15; // rsi
  int v16; // edx
  __int64 v17; // rax
  int v18; // r11d
  __int64 v19; // rdx
  unsigned int v20; // r15d
  __int64 v21; // rsi
  int v22; // esi
  __int64 v23; // rax
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  int v31; // [rsp+2Ch] [rbp-34h]

  v31 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( v31 )
  {
    v6 = a1;
    v9 = 0;
    do
    {
      while ( 1 )
      {
        v18 = v9;
        v19 = *(_QWORD *)(v6 + 32) + 40 * v9;
        if ( !*(_BYTE *)v19 )
        {
          v20 = *(_DWORD *)(v19 + 8);
          if ( v20 )
            break;
        }
LABEL_13:
        if ( v31 == (_DWORD)++v9 )
          return 0;
      }
      v21 = 24LL * v20;
      v13 = *(_DWORD *)(*(_QWORD *)(*a4 + 8LL) + v21 + 16) & 0xFFF;
      v12 = (__int16 *)(*(_QWORD *)(*a4 + 56LL) + 2LL * (*(_DWORD *)(*(_QWORD *)(*a4 + 8LL) + v21 + 16) >> 12));
      if ( (*(_BYTE *)(v19 + 3) & 0x10) != 0 )
      {
        do
        {
          if ( !v12 )
            break;
          if ( (*(_QWORD *)(a4[1] + 8LL * (v13 >> 6)) & (1LL << v13)) != 0 )
            return 1;
          v11 = *v12++;
          v13 += v11;
        }
        while ( (_WORD)v11 );
        v14 = *(_DWORD *)(*(_QWORD *)(*a5 + 8LL) + v21 + 16) & 0xFFF;
        v15 = (__int16 *)(*(_QWORD *)(*a5 + 56LL) + 2LL * (*(_DWORD *)(*(_QWORD *)(*a5 + 8LL) + v21 + 16) >> 12));
        do
        {
          if ( !v15 )
            break;
          if ( (*(_QWORD *)(a5[1] + 8LL * (v14 >> 6)) & (1LL << v14)) != 0 )
            return 1;
          v16 = *v15++;
          v14 += v16;
        }
        while ( (_WORD)v16 );
        v17 = *(unsigned int *)(a3 + 8);
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v27 = a2;
          v28 = v6;
          v30 = a3;
          sub_C8D5F0(a3, (const void *)(a3 + 16), v17 + 1, 4u, a2, a3);
          a3 = v30;
          a2 = v27;
          v6 = v28;
          v17 = *(unsigned int *)(v30 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a3 + 4 * v17) = v20;
        ++*(_DWORD *)(a3 + 8);
        goto LABEL_13;
      }
      do
      {
        if ( !v12 )
          break;
        if ( (*(_QWORD *)(a4[1] + 8LL * (v13 >> 6)) & (1LL << v13)) != 0 )
          return 1;
        v22 = *v12++;
        v13 += v22;
      }
      while ( (_WORD)v22 );
      v23 = *(unsigned int *)(a2 + 8);
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v25 = a3;
        v26 = v6;
        v29 = a2;
        sub_C8D5F0(a2, (const void *)(a2 + 16), v23 + 1, 4u, a2, a3);
        a2 = v29;
        a3 = v25;
        v6 = v26;
        v18 = v9;
        v23 = *(unsigned int *)(v29 + 8);
      }
      ++v9;
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v23) = v18;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( v31 != (_DWORD)v9 );
  }
  return 0;
}
