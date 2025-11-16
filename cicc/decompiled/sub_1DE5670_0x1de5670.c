// Function: sub_1DE5670
// Address: 0x1de5670
//
unsigned __int64 __fastcall sub_1DE5670(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r8
  _QWORD *v8; // r9
  __int64 v9; // r11
  unsigned __int64 result; // rax
  char v11; // dl
  __int64 *v12; // r10
  __int64 *v13; // r12
  __int64 v14; // rcx
  int v15; // edx
  int v16; // edi
  unsigned int v17; // eax
  __int64 v18; // rsi
  unsigned int v19; // esi
  unsigned int v20; // edi
  _QWORD *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  int v24; // eax
  __int64 *v25; // rsi
  unsigned int v26; // edi
  __int64 *v27; // rcx
  __int64 v28; // rbx
  _QWORD *v29; // rdx
  int v30; // eax
  int v31; // eax
  int v32; // esi
  int v33; // esi
  unsigned int v34; // ecx
  __int64 v35; // rdi
  int v36; // r15d
  int v37; // ecx
  int v38; // ecx
  unsigned int v39; // r15d
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 *v42; // [rsp+0h] [rbp-60h]
  __int64 *v43; // [rsp+0h] [rbp-60h]
  int v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+8h] [rbp-58h]
  __int64 v46; // [rsp+8h] [rbp-58h]
  __int64 v47; // [rsp+10h] [rbp-50h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  __int64 v49; // [rsp+20h] [rbp-40h]
  __int64 *v50; // [rsp+20h] [rbp-40h]
  __int64 v51[7]; // [rsp+28h] [rbp-38h] BYREF

  v51[0] = a2;
  v47 = a1 + 888;
  v9 = sub_1DE4FA0(a1 + 888, v51)[1];
  result = *(_QWORD *)(a3 + 8);
  if ( *(_QWORD *)(a3 + 16) != result )
  {
LABEL_2:
    v49 = v9;
    result = (unsigned __int64)sub_16CCBA0(a3, v9);
    v9 = v49;
    if ( !v11 )
      return result;
    goto LABEL_3;
  }
  v25 = (__int64 *)(result + 8LL * *(unsigned int *)(a3 + 28));
  v26 = *(_DWORD *)(a3 + 28);
  if ( (__int64 *)result == v25 )
    goto LABEL_69;
  v27 = 0;
  do
  {
    if ( v9 == *(_QWORD *)result )
      return result;
    if ( *(_QWORD *)result == -2 )
      v27 = (__int64 *)result;
    result += 8LL;
  }
  while ( v25 != (__int64 *)result );
  if ( !v27 )
  {
LABEL_69:
    if ( v26 >= *(_DWORD *)(a3 + 24) )
      goto LABEL_2;
    *(_DWORD *)(a3 + 28) = v26 + 1;
    *v25 = v9;
    ++*(_QWORD *)a3;
  }
  else
  {
    *v27 = v9;
    --*(_DWORD *)(a3 + 32);
    ++*(_QWORD *)a3;
  }
LABEL_3:
  v50 = *(__int64 **)v9;
  v48 = *(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 8);
  if ( v48 != *(_QWORD *)v9 )
  {
    while ( 1 )
    {
      v12 = *(__int64 **)(*v50 + 72);
      v13 = *(__int64 **)(*v50 + 64);
      if ( v12 != v13 )
        break;
LABEL_28:
      if ( (__int64 *)v48 == ++v50 )
        goto LABEL_29;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v23 = *v13;
        if ( !a4 )
          break;
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v14 = a4 + 16;
          v15 = 15;
        }
        else
        {
          v24 = *(_DWORD *)(a4 + 24);
          v14 = *(_QWORD *)(a4 + 16);
          if ( !v24 )
            goto LABEL_12;
          v15 = v24 - 1;
        }
        v16 = 1;
        v17 = v15 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v18 = *(_QWORD *)(v14 + 8LL * v17);
        if ( v23 == v18 )
          break;
        while ( v18 != -8 )
        {
          LODWORD(v7) = v16 + 1;
          v17 = v15 & (v16 + v17);
          v18 = *(_QWORD *)(v14 + 8LL * v17);
          if ( v23 == v18 )
            goto LABEL_8;
          ++v16;
        }
        if ( v12 == ++v13 )
          goto LABEL_28;
      }
LABEL_8:
      v19 = *(_DWORD *)(a1 + 912);
      if ( !v19 )
        break;
      LODWORD(v8) = v19 - 1;
      v7 = *(_QWORD *)(a1 + 896);
      v20 = (v19 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v21 = (_QWORD *)(v7 + 16LL * v20);
      v22 = *v21;
      if ( v23 != *v21 )
      {
        v44 = 1;
        v29 = 0;
        while ( v22 != -8 )
        {
          if ( v22 == -16 && !v29 )
            v29 = v21;
          v20 = (unsigned int)v8 & (v44 + v20);
          LODWORD(v7) = v44 + 1;
          v21 = (_QWORD *)(*(_QWORD *)(a1 + 896) + 16LL * v20);
          v22 = *v21;
          if ( v23 == *v21 )
            goto LABEL_10;
          ++v44;
        }
        if ( !v29 )
          v29 = v21;
        v30 = *(_DWORD *)(a1 + 904);
        ++*(_QWORD *)(a1 + 888);
        v31 = v30 + 1;
        if ( 4 * v31 < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 908) - v31 <= v19 >> 3 )
          {
            v43 = v12;
            v46 = v9;
            sub_1DE4DF0(v47, v19);
            v37 = *(_DWORD *)(a1 + 912);
            if ( !v37 )
            {
LABEL_76:
              ++*(_DWORD *)(a1 + 904);
              BUG();
            }
            v38 = v37 - 1;
            v8 = 0;
            v9 = v46;
            v12 = v43;
            v39 = v38 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v40 = *(_QWORD *)(a1 + 896);
            LODWORD(v7) = 1;
            v31 = *(_DWORD *)(a1 + 904) + 1;
            v29 = (_QWORD *)(v40 + 16LL * v39);
            v41 = *v29;
            if ( v23 != *v29 )
            {
              while ( v41 != -8 )
              {
                if ( v41 == -16 && !v8 )
                  v8 = v29;
                v39 = v38 & (v7 + v39);
                v29 = (_QWORD *)(v40 + 16LL * v39);
                v41 = *v29;
                if ( v23 == *v29 )
                  goto LABEL_40;
                LODWORD(v7) = v7 + 1;
              }
              goto LABEL_48;
            }
          }
          goto LABEL_40;
        }
LABEL_44:
        v42 = v12;
        v45 = v9;
        sub_1DE4DF0(v47, 2 * v19);
        v32 = *(_DWORD *)(a1 + 912);
        if ( !v32 )
          goto LABEL_76;
        v33 = v32 - 1;
        v7 = *(_QWORD *)(a1 + 896);
        v9 = v45;
        v12 = v42;
        v34 = v33 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v31 = *(_DWORD *)(a1 + 904) + 1;
        v29 = (_QWORD *)(v7 + 16LL * v34);
        v35 = *v29;
        if ( v23 != *v29 )
        {
          v36 = 1;
          v8 = 0;
          while ( v35 != -8 )
          {
            if ( !v8 && v35 == -16 )
              v8 = v29;
            v34 = v33 & (v36 + v34);
            v29 = (_QWORD *)(v7 + 16LL * v34);
            v35 = *v29;
            if ( v23 == *v29 )
              goto LABEL_40;
            ++v36;
          }
LABEL_48:
          if ( v8 )
            v29 = v8;
        }
LABEL_40:
        *(_DWORD *)(a1 + 904) = v31;
        if ( *v29 != -8 )
          --*(_DWORD *)(a1 + 908);
        *v29 = v23;
        v29[1] = 0;
LABEL_11:
        ++*(_DWORD *)(v9 + 56);
        goto LABEL_12;
      }
LABEL_10:
      if ( v21[1] != v9 )
        goto LABEL_11;
LABEL_12:
      if ( v12 == ++v13 )
        goto LABEL_28;
    }
    ++*(_QWORD *)(a1 + 888);
    goto LABEL_44;
  }
LABEL_29:
  result = *(unsigned int *)(v9 + 56);
  if ( !(_DWORD)result )
  {
    v28 = **(_QWORD **)v9;
    if ( *(_BYTE *)(v28 + 180) )
    {
      result = *(unsigned int *)(a1 + 384);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 388) )
      {
        sub_16CD150(a1 + 376, (const void *)(a1 + 392), 0, 8, v7, (int)v8);
        result = *(unsigned int *)(a1 + 384);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 376) + 8 * result) = v28;
      ++*(_DWORD *)(a1 + 384);
    }
    else
    {
      result = *(unsigned int *)(a1 + 240);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 244) )
      {
        sub_16CD150(a1 + 232, (const void *)(a1 + 248), 0, 8, v7, (int)v8);
        result = *(unsigned int *)(a1 + 240);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8 * result) = v28;
      ++*(_DWORD *)(a1 + 240);
    }
  }
  return result;
}
