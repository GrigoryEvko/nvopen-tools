// Function: sub_32546F0
// Address: 0x32546f0
//
void __fastcall sub_32546F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // r11
  unsigned int v7; // esi
  __int64 v8; // r9
  unsigned int v9; // r8d
  _QWORD *v10; // rax
  __int64 v11; // rdi
  _DWORD *v12; // rax
  __int64 v13; // r12
  void *v14; // rax
  int v15; // edi
  int v16; // edi
  __int64 v17; // r9
  unsigned int v18; // esi
  int v19; // eax
  _QWORD *v20; // rdx
  __int64 v21; // r8
  int v22; // r10d
  _QWORD *v23; // r15
  int v24; // eax
  int v25; // esi
  int v26; // esi
  __int64 v27; // r8
  _QWORD *v28; // r10
  unsigned int v29; // r15d
  int v30; // r9d
  __int64 v31; // rdi
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  int v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+20h] [rbp-40h]

  v41 = 0;
  v33 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v33 )
  {
    while ( 1 )
    {
      v4 = 0;
      v5 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v41);
      v6 = *(unsigned int *)(v5 + 16);
      if ( (_DWORD)v6 )
        break;
LABEL_22:
      if ( ++v41 == v33 )
        return;
    }
    while ( 1 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 8 * v4);
      if ( *(_QWORD *)v13 )
      {
        v7 = *(_DWORD *)(a3 + 24);
        if ( !v7 )
          goto LABEL_14;
      }
      else
      {
        if ( (*(_BYTE *)(v13 + 9) & 0x70) != 0x20 )
          goto LABEL_8;
        if ( *(char *)(v13 + 8) < 0 )
          goto LABEL_8;
        *(_BYTE *)(v13 + 8) |= 8u;
        v34 = v6;
        v37 = v5;
        v14 = sub_E807D0(*(_QWORD *)(v13 + 24));
        v5 = v37;
        v6 = v34;
        *(_QWORD *)v13 = v14;
        if ( !v14 )
          goto LABEL_8;
        v7 = *(_DWORD *)(a3 + 24);
        if ( !v7 )
        {
LABEL_14:
          ++*(_QWORD *)a3;
          goto LABEL_15;
        }
      }
      v8 = *(_QWORD *)(a3 + 8);
      v9 = (v7 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v10 = (_QWORD *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v13 != *v10 )
      {
        v39 = 1;
        v20 = 0;
        while ( v11 != -4096 )
        {
          if ( v11 == -8192 && !v20 )
            v20 = v10;
          v9 = (v7 - 1) & (v39 + v9);
          v10 = (_QWORD *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v13 == *v10 )
            goto LABEL_6;
          ++v39;
        }
        if ( !v20 )
          v20 = v10;
        v24 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v19 = v24 + 1;
        if ( 4 * v19 >= 3 * v7 )
        {
LABEL_15:
          v35 = v6;
          v38 = v5;
          sub_3254510(a3, 2 * v7);
          v15 = *(_DWORD *)(a3 + 24);
          if ( !v15 )
            goto LABEL_54;
          v16 = v15 - 1;
          v17 = *(_QWORD *)(a3 + 8);
          v5 = v38;
          v6 = v35;
          v18 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v19 = *(_DWORD *)(a3 + 16) + 1;
          v20 = (_QWORD *)(v17 + 16LL * v18);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            v22 = 1;
            v23 = 0;
            while ( v21 != -4096 )
            {
              if ( !v23 && v21 == -8192 )
                v23 = v20;
              v18 = v16 & (v22 + v18);
              v20 = (_QWORD *)(v17 + 16LL * v18);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_30;
              ++v22;
            }
            if ( v23 )
              v20 = v23;
          }
        }
        else if ( v7 - *(_DWORD *)(a3 + 20) - v19 <= v7 >> 3 )
        {
          v36 = v6;
          v40 = v5;
          sub_3254510(a3, v7);
          v25 = *(_DWORD *)(a3 + 24);
          if ( !v25 )
          {
LABEL_54:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
          v26 = v25 - 1;
          v27 = *(_QWORD *)(a3 + 8);
          v28 = 0;
          v29 = v26 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v5 = v40;
          v6 = v36;
          v30 = 1;
          v19 = *(_DWORD *)(a3 + 16) + 1;
          v20 = (_QWORD *)(v27 + 16LL * v29);
          v31 = *v20;
          if ( v13 != *v20 )
          {
            while ( v31 != -4096 )
            {
              if ( v31 == -8192 && !v28 )
                v28 = v20;
              v29 = v26 & (v30 + v29);
              v20 = (_QWORD *)(v27 + 16LL * v29);
              v31 = *v20;
              if ( v13 == *v20 )
                goto LABEL_30;
              ++v30;
            }
            if ( v28 )
              v20 = v28;
          }
        }
LABEL_30:
        *(_DWORD *)(a3 + 16) = v19;
        if ( *v20 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v20 = v13;
        v12 = v20 + 1;
        v20[1] = 0;
        goto LABEL_7;
      }
LABEL_6:
      v12 = v10 + 1;
LABEL_7:
      v12[1] = v4;
      *v12 = v41;
LABEL_8:
      if ( v6 == ++v4 )
        goto LABEL_22;
    }
  }
}
