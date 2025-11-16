// Function: sub_2B24400
// Address: 0x2b24400
//
__int64 __fastcall sub_2B24400(_BYTE **a1, unsigned __int64 a2)
{
  __int64 v2; // rsi
  _BYTE **v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rsi
  _BYTE **v6; // rdx
  _BYTE **v7; // rax
  __int64 v8; // r15
  unsigned int v9; // ebx
  unsigned int v10; // eax
  __int64 *v11; // r12
  unsigned __int64 v12; // r15
  __int64 *v13; // r13
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 *v17; // r10
  char v19; // al
  unsigned int v20; // eax
  __int64 v21; // rax
  char v22; // al
  unsigned int v23; // eax
  __int64 v24; // rax
  char v25; // al
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // ecx
  unsigned int v30; // esi
  _QWORD *v31; // rax
  unsigned __int64 *v32; // r8
  int v33; // ecx
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rax
  __int64 v37; // [rsp+0h] [rbp-A0h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 v39; // [rsp+10h] [rbp-90h]
  unsigned __int64 v40; // [rsp+18h] [rbp-88h]
  unsigned int v41; // [rsp+2Ch] [rbp-74h]
  _QWORD *v43; // [rsp+38h] [rbp-68h]
  unsigned int v44; // [rsp+40h] [rbp-60h]
  unsigned int v45; // [rsp+44h] [rbp-5Ch]
  unsigned __int64 v46; // [rsp+48h] [rbp-58h]
  __int64 *v47; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v48; // [rsp+58h] [rbp-48h]
  unsigned int v49; // [rsp+64h] [rbp-3Ch] BYREF
  __int64 v50[7]; // [rsp+68h] [rbp-38h] BYREF

  v45 = 0;
  if ( !a2 )
    return v45;
  v2 = 8 * a2;
  v3 = &a1[(unsigned __int64)v2 / 8];
  v4 = v2 >> 5;
  v5 = v2 >> 3;
  if ( v4 > 0 )
  {
    v6 = a1;
    v7 = &a1[4 * v4];
    while ( **v6 == 92 )
    {
      if ( *v6[1] != 92 )
      {
        ++v6;
        goto LABEL_5;
      }
      if ( *v6[2] != 92 )
      {
        v6 += 2;
        goto LABEL_5;
      }
      if ( *v6[3] != 92 )
      {
        v6 += 3;
        goto LABEL_5;
      }
      v6 += 4;
      if ( v6 == v7 )
      {
        v5 = v3 - v6;
        goto LABEL_86;
      }
    }
    goto LABEL_5;
  }
  v6 = a1;
LABEL_86:
  if ( v5 == 2 )
  {
LABEL_94:
    if ( **v6 == 92 )
    {
      ++v6;
      goto LABEL_89;
    }
    goto LABEL_5;
  }
  if ( v5 != 3 )
  {
    if ( v5 != 1 )
      goto LABEL_6;
LABEL_89:
    if ( **v6 == 92 )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( **v6 == 92 )
  {
    ++v6;
    goto LABEL_94;
  }
LABEL_5:
  v45 = 0;
  if ( v3 != v6 )
    return v45;
LABEL_6:
  v8 = *((_QWORD *)*a1 - 8);
  v9 = *((_DWORD *)*a1 + 20);
  v10 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL) / v9;
  v45 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL) % v9;
  v41 = v10;
  if ( v45 )
    return 0;
  if ( *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL) >= v9 )
  {
    v39 = v10;
    v46 = a2 % v10;
    if ( !v46 )
    {
      v47 = (__int64 *)a1;
      v37 = v10;
      v40 = (unsigned __int64)v10 >> 2;
      v38 = 8LL * v10;
      v43 = &a1[(unsigned __int64)v38 / 8];
      v44 = 1;
      v11 = (__int64 *)&a1[4 * v40];
      while ( 1 )
      {
        v13 = v47;
        sub_B48880(v50, v41, 0);
        if ( v40 )
        {
          while ( 1 )
          {
            v14 = *v13;
            if ( *(_QWORD *)(*v13 - 64) != v8
              || *(_BYTE *)(*(_QWORD *)(v14 + 8) + 8LL) == 18
              || !(unsigned __int8)sub_B4EFF0(
                                     *(int **)(v14 + 72),
                                     *(unsigned int *)(v14 + 80),
                                     *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL),
                                     (int *)&v49) )
            {
              goto LABEL_10;
            }
            v15 = v49 / v9;
            if ( (v50[0] & 1) != 0 )
              v50[0] = 2
                     * (((unsigned __int64)v50[0] >> 58 << 57)
                      | ~(-1LL << ((unsigned __int64)v50[0] >> 58))
                      & (~(-1LL << ((unsigned __int64)v50[0] >> 58)) & ((unsigned __int64)v50[0] >> 1) | (1LL << v15)))
                     + 1;
            else
              *(_QWORD *)(*(_QWORD *)v50[0] + 8LL * (v15 >> 6)) |= 1LL << v15;
            v16 = v13[1];
            v17 = v13 + 1;
            if ( *(_QWORD *)(v16 - 64) != v8 )
              goto LABEL_22;
            if ( *(_BYTE *)(*(_QWORD *)(v16 + 8) + 8LL) == 18 )
              goto LABEL_22;
            v19 = sub_B4EFF0(
                    *(int **)(v16 + 72),
                    *(unsigned int *)(v16 + 80),
                    *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL),
                    (int *)&v49);
            v17 = v13 + 1;
            if ( !v19 )
              goto LABEL_22;
            v20 = v49 / v9;
            if ( (v50[0] & 1) != 0 )
              v50[0] = 2
                     * (((unsigned __int64)v50[0] >> 58 << 57)
                      | ~(-1LL << ((unsigned __int64)v50[0] >> 58))
                      & (~(-1LL << ((unsigned __int64)v50[0] >> 58)) & ((unsigned __int64)v50[0] >> 1) | (1LL << v20)))
                     + 1;
            else
              *(_QWORD *)(*(_QWORD *)v50[0] + 8LL * (v20 >> 6)) |= 1LL << v20;
            v21 = v13[2];
            v17 = v13 + 2;
            if ( *(_QWORD *)(v21 - 64) != v8 )
              goto LABEL_22;
            if ( *(_BYTE *)(*(_QWORD *)(v21 + 8) + 8LL) == 18 )
              goto LABEL_22;
            v22 = sub_B4EFF0(
                    *(int **)(v21 + 72),
                    *(unsigned int *)(v21 + 80),
                    *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL),
                    (int *)&v49);
            v17 = v13 + 2;
            if ( !v22 )
              goto LABEL_22;
            v23 = v49 / v9;
            if ( (v50[0] & 1) != 0 )
              v50[0] = 2
                     * (((unsigned __int64)v50[0] >> 58 << 57)
                      | ~(-1LL << ((unsigned __int64)v50[0] >> 58))
                      & (~(-1LL << ((unsigned __int64)v50[0] >> 58)) & ((unsigned __int64)v50[0] >> 1) | (1LL << v23)))
                     + 1;
            else
              *(_QWORD *)(*(_QWORD *)v50[0] + 8LL * (v23 >> 6)) |= 1LL << v23;
            v24 = v13[3];
            v17 = v13 + 3;
            if ( *(_QWORD *)(v24 - 64) != v8
              || *(_BYTE *)(*(_QWORD *)(v24 + 8) + 8LL) == 18
              || (v25 = sub_B4EFF0(
                          *(int **)(v24 + 72),
                          *(unsigned int *)(v24 + 80),
                          *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL),
                          (int *)&v49),
                  v17 = v13 + 3,
                  !v25) )
            {
LABEL_22:
              v13 = v17;
              goto LABEL_10;
            }
            v26 = v49 / v9;
            if ( (v50[0] & 1) != 0 )
              v50[0] = 2
                     * (((unsigned __int64)v50[0] >> 58 << 57)
                      | ~(-1LL << ((unsigned __int64)v50[0] >> 58))
                      & (~(-1LL << ((unsigned __int64)v50[0] >> 58)) & ((unsigned __int64)v50[0] >> 1) | (1LL << v26)))
                     + 1;
            else
              *(_QWORD *)(*(_QWORD *)v50[0] + 8LL * (v26 >> 6)) |= 1LL << v26;
            v13 += 4;
            if ( v13 == v11 )
            {
              v27 = v43 - v13;
              goto LABEL_41;
            }
          }
        }
        v27 = v37;
LABEL_41:
        if ( v27 == 2 )
          goto LABEL_75;
        if ( v27 == 3 )
          break;
        if ( v27 != 1 )
          goto LABEL_11;
LABEL_44:
        v28 = *v13;
        if ( *(_QWORD *)(*v13 - 64) == v8
          && *(_BYTE *)(*(_QWORD *)(v28 + 8) + 8LL) != 18
          && (unsigned __int8)sub_B4EFF0(
                                *(int **)(v28 + 72),
                                *(unsigned int *)(v28 + 80),
                                *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL),
                                (int *)&v49) )
        {
          sub_2B0D980(v50, v49 / v9);
          goto LABEL_11;
        }
LABEL_10:
        if ( v43 != v13 )
        {
          v32 = (unsigned __int64 *)v50[0];
          if ( (v50[0] & 1) != 0 )
            return v45;
LABEL_57:
          if ( v32 )
          {
            if ( (unsigned __int64 *)*v32 != v32 + 2 )
            {
              v48 = v32;
              _libc_free(*v32);
              v32 = v48;
            }
            j_j___libc_free_0((unsigned __int64)v32);
          }
          return v45;
        }
LABEL_11:
        v12 = v50[0];
        if ( (v50[0] & 1) != 0 )
        {
          if ( (~(-1LL << ((unsigned __int64)v50[0] >> 58)) & ((unsigned __int64)v50[0] >> 1)) != (1LL << ((unsigned __int64)v50[0] >> 58))
                                                                                                - 1 )
            return v45;
        }
        else
        {
          v29 = *(_DWORD *)(v50[0] + 64);
          v30 = v29 >> 6;
          if ( v29 >> 6 )
          {
            v31 = *(_QWORD **)v50[0];
            while ( *v31 == -1 )
            {
              if ( (_QWORD *)(*(_QWORD *)v50[0] + 8LL * (v30 - 1) + 8) == ++v31 )
                goto LABEL_61;
            }
LABEL_56:
            v32 = (unsigned __int64 *)v50[0];
            goto LABEL_57;
          }
LABEL_61:
          v33 = v29 & 0x3F;
          if ( v33 && *(_QWORD *)(*(_QWORD *)v50[0] + 8LL * v30) != (1LL << v33) - 1 )
            goto LABEL_56;
          if ( v50[0] )
          {
            if ( *(_QWORD *)v50[0] != v50[0] + 16 )
              _libc_free(*(_QWORD *)v50[0]);
            j_j___libc_free_0(v12);
          }
        }
        v46 += v39;
        v47 = (__int64 *)((char *)v47 + v38);
        v11 = (__int64 *)((char *)v11 + v38);
        v43 = (_QWORD *)((char *)v43 + v38);
        if ( a2 == v46 )
          return v44;
        ++v44;
        v8 = *(_QWORD *)(*v47 - 64);
      }
      v34 = *v13;
      if ( *(_QWORD *)(*v13 - 64) != v8
        || *(_BYTE *)(*(_QWORD *)(v34 + 8) + 8LL) == 18
        || !(unsigned __int8)sub_B4EFF0(
                               *(int **)(v34 + 72),
                               *(unsigned int *)(v34 + 80),
                               *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL),
                               (int *)&v49) )
      {
        goto LABEL_10;
      }
      v35 = v49 / v9;
      if ( (v50[0] & 1) != 0 )
        v50[0] = 2
               * (((unsigned __int64)v50[0] >> 58 << 57)
                | ~(-1LL << ((unsigned __int64)v50[0] >> 58))
                & (~(-1LL << ((unsigned __int64)v50[0] >> 58)) & ((unsigned __int64)v50[0] >> 1) | (1LL << v35)))
               + 1;
      else
        *(_QWORD *)(*(_QWORD *)v50[0] + 8LL * (v35 >> 6)) |= 1LL << v35;
      ++v13;
LABEL_75:
      v36 = *v13;
      if ( *(_QWORD *)(*v13 - 64) != v8
        || *(_BYTE *)(*(_QWORD *)(v36 + 8) + 8LL) == 18
        || !(unsigned __int8)sub_B4EFF0(
                               *(int **)(v36 + 72),
                               *(unsigned int *)(v36 + 80),
                               *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL),
                               (int *)&v49) )
      {
        goto LABEL_10;
      }
      ++v13;
      sub_2B0D980(v50, v49 / v9);
      goto LABEL_44;
    }
  }
  return v45;
}
