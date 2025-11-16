// Function: sub_1DF98C0
// Address: 0x1df98c0
//
__int64 __fastcall sub_1DF98C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v9; // rbx
  __int64 v10; // r14
  int v11; // edx
  int v12; // edx
  __int64 v13; // rdi
  int v14; // r9d
  unsigned int v15; // r8d
  int v16; // esi
  unsigned int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // ecx
  int *v20; // rdx
  int v21; // edi
  __int64 v22; // r13
  int v23; // edx
  int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // edx
  int v27; // esi
  int v28; // r8d
  unsigned int v29; // esi
  __int64 v30; // r8
  unsigned int v31; // r9d
  _DWORD *v32; // rdx
  int v33; // ecx
  int *v34; // r13
  int v35; // edi
  int v36; // edi
  int v37; // edi
  int v38; // ecx
  __int64 v39; // [rsp+8h] [rbp-68h]
  __int64 v40; // [rsp+8h] [rbp-68h]
  __int64 v41; // [rsp+10h] [rbp-60h]
  int v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+10h] [rbp-60h]
  __int64 v44; // [rsp+10h] [rbp-60h]
  int v45; // [rsp+20h] [rbp-50h]
  __int64 v46; // [rsp+20h] [rbp-50h]
  _DWORD *v47; // [rsp+20h] [rbp-50h]
  _DWORD *v48; // [rsp+20h] [rbp-50h]
  __int64 v49; // [rsp+28h] [rbp-48h]
  unsigned int v50; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v51[7]; // [rsp+38h] [rbp-38h] BYREF

  result = a2 + 24;
  v5 = *(_QWORD *)(a2 + 32);
  v49 = a2 + 24;
  if ( v5 != a2 + 24 )
  {
    while ( 1 )
    {
      result = *(unsigned int *)(v5 + 40);
      if ( (_DWORD)result )
        break;
LABEL_25:
      if ( (*(_BYTE *)v5 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v5 + 46) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
      }
      v5 = *(_QWORD *)(v5 + 8);
      if ( v49 == v5 )
        return result;
    }
    result = (unsigned int)(result - 1);
    v9 = 0;
    v10 = 8 * (5 * result + 5);
    while ( 1 )
    {
      v22 = v9 + *(_QWORD *)(v5 + 32);
      if ( *(_BYTE *)v22 )
        goto LABEL_9;
      result = *(unsigned int *)(v22 + 8);
      if ( (int)result >= 0 )
        goto LABEL_9;
      if ( (*(_BYTE *)(v22 + 3) & 0x10) != 0 )
        break;
LABEL_4:
      v11 = *(_DWORD *)(a4 + 24);
      if ( v11 )
      {
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a4 + 8);
        v14 = 1;
        v15 = v12 & (37 * result);
        v16 = *(_DWORD *)(v13 + 4LL * v15);
        if ( v16 == (_DWORD)result )
        {
LABEL_6:
          v17 = *(_DWORD *)(a3 + 24);
          v50 = result;
          if ( v17 )
          {
            v18 = *(_QWORD *)(a3 + 8);
            v19 = (v17 - 1) & (37 * result);
            v20 = (int *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( *v20 == (_DWORD)result )
            {
LABEL_8:
              *((_QWORD *)v20 + 1) = v5;
              goto LABEL_9;
            }
            v45 = 1;
            v34 = 0;
            while ( v21 != -1 )
            {
              if ( !v34 && v21 == -2 )
                v34 = v20;
              v19 = (v17 - 1) & (v45 + v19);
              v20 = (int *)(v18 + 16LL * v19);
              v21 = *v20;
              if ( (_DWORD)result == *v20 )
                goto LABEL_8;
              ++v45;
            }
            v35 = *(_DWORD *)(a3 + 16);
            if ( v34 )
              v20 = v34;
            ++*(_QWORD *)a3;
            v36 = v35 + 1;
            if ( 4 * v36 < 3 * v17 )
            {
              if ( v17 - *(_DWORD *)(a3 + 20) - v36 > v17 >> 3 )
              {
LABEL_37:
                *(_DWORD *)(a3 + 16) = v36;
                if ( *v20 != -1 )
                  --*(_DWORD *)(a3 + 20);
                *v20 = result;
                *((_QWORD *)v20 + 1) = 0;
                goto LABEL_8;
              }
              v41 = a1;
LABEL_42:
              v46 = a3;
              sub_1DF4FB0(a3, v17);
              sub_1DF9340(v46, (int *)&v50, v51);
              a3 = v46;
              v20 = (int *)v51[0];
              result = v50;
              a1 = v41;
              v36 = *(_DWORD *)(v46 + 16) + 1;
              goto LABEL_37;
            }
          }
          else
          {
            ++*(_QWORD *)a3;
          }
          v41 = a1;
          v17 *= 2;
          goto LABEL_42;
        }
        while ( v16 != -1 )
        {
          v15 = v12 & (v14 + v15);
          v16 = *(_DWORD *)(v13 + 4LL * v15);
          if ( v16 == (_DWORD)result )
            goto LABEL_6;
          ++v14;
        }
        v9 += 40;
        if ( v10 == v9 )
          goto LABEL_25;
      }
      else
      {
LABEL_9:
        v9 += 40;
        if ( v10 == v9 )
          goto LABEL_25;
      }
    }
    v23 = *(_DWORD *)(a1 + 80);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 64);
      v26 = (v23 - 1) & (37 * result);
      v27 = *(_DWORD *)(v25 + 8LL * v26);
      if ( (_DWORD)result == v27 )
        goto LABEL_9;
      v28 = 1;
      while ( v27 != -1 )
      {
        v26 = v24 & (v28 + v26);
        v27 = *(_DWORD *)(v25 + 8LL * v26);
        if ( (_DWORD)result == v27 )
          goto LABEL_9;
        ++v28;
      }
    }
    v29 = *(_DWORD *)(a4 + 24);
    v50 = *(_DWORD *)(v22 + 8);
    if ( v29 )
    {
      v30 = *(_QWORD *)(a4 + 8);
      v31 = (v29 - 1) & (37 * result);
      v32 = (_DWORD *)(v30 + 4LL * v31);
      v33 = *v32;
      if ( (_DWORD)result == *v32 )
      {
LABEL_19:
        if ( (*(_BYTE *)(v22 + 3) & 0x10) != 0 )
          goto LABEL_9;
        result = *(unsigned int *)(v22 + 8);
        goto LABEL_4;
      }
      v42 = 1;
      v47 = 0;
      while ( v33 != -1 )
      {
        if ( v33 != -2 || v47 )
          v32 = v47;
        v31 = (v29 - 1) & (v42 + v31);
        v33 = *(_DWORD *)(v30 + 4LL * v31);
        if ( (_DWORD)result == v33 )
          goto LABEL_19;
        v47 = v32;
        v32 = (_DWORD *)(v30 + 4LL * v31);
        ++v42;
      }
      if ( v47 )
        v32 = v47;
      v37 = *(_DWORD *)(a4 + 16);
      ++*(_QWORD *)a4;
      v38 = v37 + 1;
      v48 = v32;
      if ( 4 * (v37 + 1) < 3 * v29 )
      {
        if ( v29 - *(_DWORD *)(a4 + 20) - v38 <= v29 >> 3 )
        {
          v40 = a3;
          v44 = a1;
          sub_136B240(a4, v29);
          sub_1DF91F0(a4, (int *)&v50, v51);
          a3 = v40;
          a1 = v44;
          v48 = (_DWORD *)v51[0];
          v38 = *(_DWORD *)(a4 + 16) + 1;
          result = v50;
        }
        goto LABEL_50;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    v39 = a3;
    v43 = a1;
    sub_136B240(a4, 2 * v29);
    sub_1DF91F0(a4, (int *)&v50, v51);
    a1 = v43;
    a3 = v39;
    v48 = (_DWORD *)v51[0];
    v38 = *(_DWORD *)(a4 + 16) + 1;
    result = v50;
LABEL_50:
    *(_DWORD *)(a4 + 16) = v38;
    if ( *v48 != -1 )
      --*(_DWORD *)(a4 + 20);
    *v48 = result;
    goto LABEL_19;
  }
  return result;
}
