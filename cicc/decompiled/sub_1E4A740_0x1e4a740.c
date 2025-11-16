// Function: sub_1E4A740
// Address: 0x1e4a740
//
__int64 __fastcall sub_1E4A740(__int64 a1, unsigned __int64 a2, int a3, int a4, int a5)
{
  int v5; // eax
  int v7; // ecx
  unsigned int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // edx
  int *v11; // r12
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // r14
  void (*v18)(void); // rax
  __int64 v19; // rsi
  void (*v20)(void); // rax
  __int64 v21; // rsi
  int v23; // r10d
  int *v24; // r9
  int v25; // eax
  int v26; // edx
  int v27; // eax
  int v28; // eax
  __int64 v29; // r8
  unsigned int v30; // edi
  int v31; // esi
  int v32; // r10d
  int *v33; // r9
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  int v37; // r10d
  unsigned int v38; // edi
  int v39; // eax
  __int64 *v40; // rax
  unsigned __int64 *v41; // rdx
  unsigned __int64 v42; // rcx
  int v43; // eax
  char v44; // [rsp+3h] [rbp-6Dh]
  int v45; // [rsp+4h] [rbp-6Ch]
  int v46; // [rsp+8h] [rbp-68h]
  int v48; // [rsp+10h] [rbp-60h]
  int v49; // [rsp+14h] [rbp-5Ch]
  unsigned __int64 v50; // [rsp+18h] [rbp-58h] BYREF
  int v51; // [rsp+2Ch] [rbp-44h] BYREF
  unsigned __int64 v52; // [rsp+30h] [rbp-40h] BYREF
  int v53; // [rsp+38h] [rbp-38h]

  v5 = a3;
  v50 = a2;
  if ( a3 > a4 )
  {
    v7 = a4 - 1;
    v44 = 0;
  }
  else
  {
    v7 = a4 + 1;
    v44 = 1;
  }
  v45 = v7;
  v51 = a3;
  if ( v7 == a3 )
    return 0;
  v46 = 37 * a5;
  while ( 1 )
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 160) + 8LL) = 0;
    v49 = *(_DWORD *)(a1 + 128) + (v5 - *(_DWORD *)(a1 + 128)) % a5;
    if ( v49 <= *(_DWORD *)(a1 + 132) )
    {
      v48 = 37 * v49;
      do
      {
        v8 = *(_DWORD *)(a1 + 24);
        if ( v8 )
        {
          v9 = *(_QWORD *)(a1 + 8);
          v10 = (v8 - 1) & v48;
          v11 = (int *)(v9 + 88LL * v10);
          v12 = *v11;
          if ( *v11 == v49 )
            goto LABEL_9;
          v23 = 1;
          v24 = 0;
          while ( v12 != 0x7FFFFFFF )
          {
            if ( v12 == 0x80000000 && !v24 )
              v24 = v11;
            v10 = (v8 - 1) & (v23 + v10);
            v11 = (int *)(v9 + 88LL * v10);
            v12 = *v11;
            if ( *v11 == v49 )
              goto LABEL_9;
            ++v23;
          }
          v25 = *(_DWORD *)(a1 + 16);
          if ( v24 )
            v11 = v24;
          ++*(_QWORD *)a1;
          v26 = v25 + 1;
          if ( 4 * (v25 + 1) < 3 * v8 )
          {
            if ( v8 - *(_DWORD *)(a1 + 20) - v26 > v8 >> 3 )
              goto LABEL_32;
            sub_1E4A230(a1, v8);
            v34 = *(_DWORD *)(a1 + 24);
            if ( !v34 )
            {
LABEL_72:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v35 = v34 - 1;
            v36 = *(_QWORD *)(a1 + 8);
            v37 = 1;
            v33 = 0;
            v38 = (v34 - 1) & v48;
            v26 = *(_DWORD *)(a1 + 16) + 1;
            v11 = (int *)(v36 + 88LL * v38);
            v39 = *v11;
            if ( *v11 == v49 )
              goto LABEL_32;
            while ( v39 != 0x7FFFFFFF )
            {
              if ( v39 == 0x80000000 && !v33 )
                v33 = v11;
              v38 = v35 & (v37 + v38);
              v11 = (int *)(v36 + 88LL * v38);
              v39 = *v11;
              if ( *v11 == v49 )
                goto LABEL_32;
              ++v37;
            }
            goto LABEL_40;
          }
        }
        else
        {
          ++*(_QWORD *)a1;
        }
        sub_1E4A230(a1, 2 * v8);
        v27 = *(_DWORD *)(a1 + 24);
        if ( !v27 )
          goto LABEL_72;
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 8);
        v30 = v28 & v48;
        v11 = (int *)(v29 + 88LL * (v28 & (unsigned int)v48));
        v26 = *(_DWORD *)(a1 + 16) + 1;
        v31 = *v11;
        if ( *v11 == v49 )
          goto LABEL_32;
        v32 = 1;
        v33 = 0;
        while ( v31 != 0x7FFFFFFF )
        {
          if ( v31 == 0x80000000 && !v33 )
            v33 = v11;
          v30 = v28 & (v32 + v30);
          v11 = (int *)(v29 + 88LL * v30);
          v31 = *v11;
          if ( *v11 == v49 )
            goto LABEL_32;
          ++v32;
        }
LABEL_40:
        if ( v33 )
          v11 = v33;
LABEL_32:
        *(_DWORD *)(a1 + 16) = v26;
        if ( *v11 != 0x7FFFFFFF )
          --*(_DWORD *)(a1 + 20);
        *((_QWORD *)v11 + 1) = 0;
        *((_QWORD *)v11 + 2) = 0;
        *v11 = v49;
        *((_QWORD *)v11 + 3) = 0;
        *((_QWORD *)v11 + 4) = 0;
        *((_QWORD *)v11 + 5) = 0;
        *((_QWORD *)v11 + 6) = 0;
        *((_QWORD *)v11 + 7) = 0;
        *((_QWORD *)v11 + 8) = 0;
        *((_QWORD *)v11 + 9) = 0;
        *((_QWORD *)v11 + 10) = 0;
        sub_1E47CF0((__int64 *)v11 + 1, 0);
LABEL_9:
        v13 = *((_QWORD *)v11 + 3);
        v14 = *((_QWORD *)v11 + 5);
        v15 = *((_QWORD *)v11 + 6);
        v16 = *((_QWORD *)v11 + 7);
        v17 = v13;
        while ( v16 != v17 )
        {
          while ( 1 )
          {
            v18 = *(void (**)(void))(**(_QWORD **)(a1 + 144) + 40LL);
            if ( (char *)v18 != (char *)sub_1D00B00 )
              v18();
            v19 = *(_QWORD *)(*(_QWORD *)v17 + 8LL);
            if ( **(_WORD **)(v19 + 16) > 0xFu )
              sub_20E8EF0(*(_QWORD *)(a1 + 160), v19);
            v17 += 8;
            if ( v14 != v17 )
              break;
            v17 = *(_QWORD *)(v15 + 8);
            v15 += 8;
            v14 = v17 + 512;
            if ( v16 == v17 )
              goto LABEL_17;
          }
        }
LABEL_17:
        v49 += a5;
        v48 += v46;
      }
      while ( *(_DWORD *)(a1 + 132) >= v49 );
    }
    v20 = *(void (**)(void))(**(_QWORD **)(a1 + 144) + 40LL);
    if ( (char *)v20 != (char *)sub_1D00B00 )
      v20();
    v21 = *(_QWORD *)(v50 + 8);
    if ( **(_WORD **)(v21 + 16) <= 0xFu || (unsigned __int8)sub_20E8BB0(*(_QWORD *)(a1 + 160), v21) )
      break;
    v5 = v51 + 1;
    if ( !v44 )
      v5 = v51 - 1;
    v51 = v5;
    if ( v45 == v5 )
      return 0;
  }
  v40 = (__int64 *)sub_1E4A590(a1, &v51);
  v41 = (unsigned __int64 *)v40[7];
  if ( v41 == (unsigned __int64 *)(v40[9] - 8) )
  {
    sub_1E48ED0(v40 + 1, &v50);
    v42 = v50;
  }
  else
  {
    v42 = v50;
    if ( v41 )
    {
      *v41 = v50;
      v41 = (unsigned __int64 *)v40[7];
    }
    v40[7] = (__int64)(v41 + 1);
  }
  v52 = v42;
  v53 = v51;
  sub_1E47BD0((_QWORD *)(a1 + 32), &v52);
  v43 = v51;
  if ( *(_DWORD *)(a1 + 132) < v51 )
    *(_DWORD *)(a1 + 132) = v51;
  if ( v43 < *(_DWORD *)(a1 + 128) )
    *(_DWORD *)(a1 + 128) = v43;
  return 1;
}
