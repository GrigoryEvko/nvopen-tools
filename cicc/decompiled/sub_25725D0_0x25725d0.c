// Function: sub_25725D0
// Address: 0x25725d0
//
__int64 __fastcall sub_25725D0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // r12d
  unsigned int v9; // esi
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 *v12; // r9
  int v13; // r15d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r11
  int v17; // eax
  __int64 *v18; // r12
  __int64 v19; // rcx
  __int64 v20; // r9
  _QWORD *v21; // rdi
  int v22; // r11d
  unsigned int v23; // edx
  _QWORD *v24; // rax
  __int64 v25; // r8
  _DWORD *v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rdx
  _BYTE *v29; // r15
  unsigned int v30; // esi
  int v31; // eax
  __int64 v32; // rcx
  int v33; // edx
  __int64 v34; // rsi
  unsigned int v35; // eax
  __int64 v36; // r8
  int v37; // eax
  int v38; // edx
  __int64 v39; // rax
  int v40; // eax
  int v41; // eax
  int v42; // edx
  int v43; // eax
  int v44; // r10d
  _QWORD *v45; // r9
  __int64 v46; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v47[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) != v3 )
    *(_QWORD *)(a1 + 72) = v3;
  result = *(_QWORD *)(a1 + 88);
  if ( *(_QWORD *)(a1 + 96) != result )
  {
    while ( 1 )
    {
      sub_2572490(a1);
      v5 = *(_QWORD *)(a1 + 96);
      v6 = *(_QWORD *)(v5 - 32);
      v7 = v5 - 32;
      v46 = v6;
      v8 = *(_DWORD *)(v7 + 24);
      *(_QWORD *)(a1 + 96) = v7;
      if ( *(_QWORD *)(a1 + 88) != v7 && *(_DWORD *)(v7 - 8) > v8 )
        *(_DWORD *)(v7 - 8) = v8;
      v9 = *(_DWORD *)(a1 + 32);
      if ( !v9 )
        break;
      v10 = v46;
      v11 = *(_QWORD *)(a1 + 16);
      v12 = 0;
      v13 = 1;
      v14 = (v9 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
      v15 = (__int64 *)(v11 + 16LL * v14);
      v16 = *v15;
      if ( v46 == *v15 )
      {
LABEL_9:
        v17 = *((_DWORD *)v15 + 2);
        goto LABEL_10;
      }
      while ( v16 != -4096 )
      {
        if ( !v12 && v16 == -8192 )
          v12 = v15;
        v14 = (v9 - 1) & (v13 + v14);
        v15 = (__int64 *)(v11 + 16LL * v14);
        v16 = *v15;
        if ( v46 == *v15 )
          goto LABEL_9;
        ++v13;
      }
      if ( !v12 )
        v12 = v15;
      v41 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v42 = v41 + 1;
      v47[0] = v12;
      if ( 4 * (v41 + 1) >= 3 * v9 )
        goto LABEL_55;
      if ( v9 - *(_DWORD *)(a1 + 28) - v42 <= v9 >> 3 )
        goto LABEL_56;
LABEL_51:
      *(_DWORD *)(a1 + 24) = v42;
      if ( *v12 != -4096 )
        --*(_DWORD *)(a1 + 28);
      *v12 = v10;
      v17 = 0;
      *((_DWORD *)v12 + 2) = 0;
LABEL_10:
      if ( v8 == v17 )
      {
        v18 = *(__int64 **)(a1 + 72);
        while ( 1 )
        {
          v27 = *(_QWORD *)(a1 + 48);
          v28 = (_QWORD *)(v27 - 8);
          if ( *(__int64 **)(a1 + 80) == v18 )
          {
            sub_9319A0(v1, v18, v28);
            v29 = *(_BYTE **)(a1 + 72);
            v18 = (__int64 *)(v29 - 8);
            v28 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
          }
          else
          {
            if ( v18 )
            {
              *v18 = *(_QWORD *)(v27 - 8);
              v18 = *(__int64 **)(a1 + 72);
              v28 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
            }
            v29 = v18 + 1;
            *(_QWORD *)(a1 + 72) = v18 + 1;
          }
          v30 = *(_DWORD *)(a1 + 32);
          *(_QWORD *)(a1 + 48) = v28;
          if ( !v30 )
            break;
          v19 = *((_QWORD *)v29 - 1);
          v20 = *(_QWORD *)(a1 + 16);
          v21 = 0;
          v22 = 1;
          v23 = (v30 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v24 = (_QWORD *)(v20 + 16LL * v23);
          v25 = *v24;
          if ( *v24 != v19 )
          {
            while ( v25 != -4096 )
            {
              if ( !v21 && v25 == -8192 )
                v21 = v24;
              v23 = (v30 - 1) & (v22 + v23);
              v24 = (_QWORD *)(v20 + 16LL * v23);
              v25 = *v24;
              if ( v19 == *v24 )
                goto LABEL_15;
              ++v22;
            }
            if ( !v21 )
              v21 = v24;
            v40 = *(_DWORD *)(a1 + 24);
            ++*(_QWORD *)(a1 + 8);
            v38 = v40 + 1;
            v47[0] = v21;
            if ( 4 * (v40 + 1) < 3 * v30 )
            {
              if ( v30 - *(_DWORD *)(a1 + 28) - v38 <= v30 >> 3 )
              {
                sub_B23080(a1 + 8, v30);
                sub_B1C700(a1 + 8, v18, v47);
                v21 = (_QWORD *)v47[0];
                v38 = *(_DWORD *)(a1 + 24) + 1;
              }
LABEL_26:
              *(_DWORD *)(a1 + 24) = v38;
              if ( *v21 != -4096 )
                --*(_DWORD *)(a1 + 28);
              v39 = *((_QWORD *)v29 - 1);
              *((_DWORD *)v21 + 2) = 0;
              *v21 = v39;
              v26 = v21 + 1;
              goto LABEL_16;
            }
LABEL_23:
            sub_B23080(a1 + 8, 2 * v30);
            v31 = *(_DWORD *)(a1 + 32);
            if ( v31 )
            {
              v32 = *((_QWORD *)v29 - 1);
              v33 = v31 - 1;
              v34 = *(_QWORD *)(a1 + 16);
              v35 = (v31 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
              v21 = (_QWORD *)(v34 + 16LL * v35);
              v36 = *v21;
              if ( *v21 == v32 )
              {
LABEL_25:
                v37 = *(_DWORD *)(a1 + 24);
                v47[0] = v21;
                v38 = v37 + 1;
              }
              else
              {
                v44 = 1;
                v45 = 0;
                while ( v36 != -4096 )
                {
                  if ( v36 == -8192 && !v45 )
                    v45 = v21;
                  v35 = v33 & (v44 + v35);
                  v21 = (_QWORD *)(v34 + 16LL * v35);
                  v36 = *v21;
                  if ( v32 == *v21 )
                    goto LABEL_25;
                  ++v44;
                }
                if ( !v45 )
                  v45 = v21;
                v38 = *(_DWORD *)(a1 + 24) + 1;
                v47[0] = v45;
                v21 = v45;
              }
            }
            else
            {
              v43 = *(_DWORD *)(a1 + 24);
              v47[0] = 0;
              v21 = 0;
              v38 = v43 + 1;
            }
            goto LABEL_26;
          }
LABEL_15:
          v26 = v24 + 1;
LABEL_16:
          *v26 = -1;
          v18 = *(__int64 **)(a1 + 72);
          result = v46;
          if ( *(v18 - 1) == v46 )
            return result;
        }
        ++*(_QWORD *)(a1 + 8);
        v47[0] = 0;
        goto LABEL_23;
      }
      result = *(_QWORD *)(a1 + 96);
      if ( *(_QWORD *)(a1 + 88) == result )
        return result;
    }
    ++*(_QWORD *)(a1 + 8);
    v47[0] = 0;
LABEL_55:
    v9 *= 2;
LABEL_56:
    sub_B23080(a1 + 8, v9);
    sub_B1C700(a1 + 8, &v46, v47);
    v10 = v46;
    v12 = (__int64 *)v47[0];
    v42 = *(_DWORD *)(a1 + 24) + 1;
    goto LABEL_51;
  }
  return result;
}
