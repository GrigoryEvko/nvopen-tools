// Function: sub_1DE7010
// Address: 0x1de7010
//
__int64 __fastcall sub_1DE7010(__int64 a1, __int64 a2, _QWORD **a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rbx
  unsigned int v9; // r14d
  __int64 v10; // rsi
  int v11; // ecx
  int v12; // r8d
  unsigned int v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // esi
  int v16; // r9d
  __int64 v17; // r10
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r8
  _QWORD **v21; // rax
  unsigned int v22; // eax
  unsigned int v23; // edx
  bool v24; // cc
  __int64 v25; // rdx
  bool v26; // zf
  int v27; // ecx
  __int64 *v28; // rdi
  int v29; // eax
  int v30; // eax
  __int64 v31; // rax
  int v33; // ecx
  int v34; // ecx
  unsigned int v35; // r10d
  __int64 v36; // rsi
  __int64 *v37; // r11
  int v38; // [rsp+10h] [rbp-60h]
  __int64 *v41; // [rsp+28h] [rbp-48h]
  __int64 v42; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v43; // [rsp+38h] [rbp-38h] BYREF

  v5 = *(__int64 **)(a2 + 88);
  v41 = *(__int64 **)(a2 + 96);
  if ( v41 != v5 )
  {
    v9 = 0x80000000;
    while ( 1 )
    {
      v25 = *v5;
      v26 = *(_BYTE *)(*v5 + 180) == 0;
      v42 = *v5;
      if ( !v26 )
        goto LABEL_8;
      if ( a4 )
      {
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v10 = a4 + 16;
          v11 = 15;
        }
        else
        {
          v27 = *(_DWORD *)(a4 + 24);
          v10 = *(_QWORD *)(a4 + 16);
          if ( !v27 )
            goto LABEL_8;
          v11 = v27 - 1;
        }
        v12 = 1;
        v13 = v11 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v14 = *(_QWORD *)(v10 + 8LL * v13);
        if ( v25 != v14 )
        {
          while ( v14 != -8 )
          {
            v13 = v11 & (v12 + v13);
            v14 = *(_QWORD *)(v10 + 8LL * v13);
            if ( v25 == v14 )
              goto LABEL_5;
            ++v12;
          }
LABEL_8:
          v22 = sub_1DF1780(*(_QWORD *)(a1 + 560), a2, v25);
          v23 = v9 - v22;
          v24 = v22 <= v9;
          v9 = 0;
          if ( v24 )
            v9 = v23;
          goto LABEL_10;
        }
      }
LABEL_5:
      v15 = *(_DWORD *)(a1 + 912);
      v16 = a1 + 888;
      if ( !v15 )
        break;
      v17 = *(_QWORD *)(a1 + 896);
      v18 = (v15 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v19 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( v25 != *v19 )
      {
        v38 = 1;
        v28 = 0;
        while ( v20 != -8 )
        {
          if ( v20 == -16 && !v28 )
            v28 = v19;
          v18 = (v15 - 1) & (v38 + v18);
          v19 = (__int64 *)(v17 + 16LL * v18);
          v20 = *v19;
          if ( v25 == *v19 )
            goto LABEL_7;
          ++v38;
        }
        if ( !v28 )
          v28 = v19;
        v29 = *(_DWORD *)(a1 + 904);
        ++*(_QWORD *)(a1 + 888);
        v30 = v29 + 1;
        if ( 4 * v30 < 3 * v15 )
        {
          LODWORD(v20) = v15 >> 3;
          if ( v15 - *(_DWORD *)(a1 + 908) - v30 <= v15 >> 3 )
          {
            sub_1DE4DF0(a1 + 888, v15);
            sub_1DE30F0(a1 + 888, &v42, &v43);
            v28 = v43;
            v25 = v42;
            v30 = *(_DWORD *)(a1 + 904) + 1;
          }
LABEL_23:
          *(_DWORD *)(a1 + 904) = v30;
          if ( *v28 != -8 )
            --*(_DWORD *)(a1 + 908);
          v28[1] = 0;
          v21 = 0;
          *v28 = v25;
          v25 = v42;
          goto LABEL_26;
        }
LABEL_36:
        sub_1DE4DF0(a1 + 888, 2 * v15);
        v33 = *(_DWORD *)(a1 + 912);
        if ( !v33 )
        {
          ++*(_DWORD *)(a1 + 904);
          BUG();
        }
        v25 = v42;
        v34 = v33 - 1;
        v20 = *(_QWORD *)(a1 + 896);
        v35 = v34 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v30 = *(_DWORD *)(a1 + 904) + 1;
        v28 = (__int64 *)(v20 + 16LL * v35);
        v36 = *v28;
        if ( *v28 != v42 )
        {
          v16 = 1;
          v37 = 0;
          while ( v36 != -8 )
          {
            if ( v36 == -16 && !v37 )
              v37 = v28;
            v35 = v34 & (v16 + v35);
            v28 = (__int64 *)(v20 + 16LL * v35);
            v36 = *v28;
            if ( v42 == *v28 )
              goto LABEL_23;
            ++v16;
          }
          if ( v37 )
            v28 = v37;
        }
        goto LABEL_23;
      }
LABEL_7:
      v21 = (_QWORD **)v19[1];
      if ( v21 == a3 )
        goto LABEL_8;
LABEL_26:
      if ( **v21 == v25 )
      {
        v31 = *(unsigned int *)(a5 + 8);
        if ( (unsigned int)v31 >= *(_DWORD *)(a5 + 12) )
        {
          sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v20, v16);
          v31 = *(unsigned int *)(a5 + 8);
        }
        ++v5;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v31) = v42;
        ++*(_DWORD *)(a5 + 8);
        if ( v41 == v5 )
          return v9;
      }
      else
      {
LABEL_10:
        if ( v41 == ++v5 )
          return v9;
      }
    }
    ++*(_QWORD *)(a1 + 888);
    goto LABEL_36;
  }
  return 0x80000000;
}
