// Function: sub_15EF740
// Address: 0x15ef740
//
__int64 __fastcall sub_15EF740(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 v4; // r15
  _BYTE *v5; // rsi
  _BYTE *v6; // rdi
  size_t v7; // rdx
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // r13
  unsigned int v11; // eax
  int v12; // eax
  unsigned int v13; // esi
  char v14; // al
  char v15; // al
  __int64 v16; // rbx
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rdx
  unsigned __int64 v20; // rbx
  __int64 v21; // rdx
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // r14
  _QWORD *v26; // r13
  unsigned __int64 v27; // r15
  _QWORD *v28; // r12
  __int64 result; // rax
  __int64 v30; // r13
  __int64 v31; // r14
  _BYTE *v32; // rdi
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  unsigned __int64 v36; // [rsp+18h] [rbp-68h]
  _QWORD v37[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v38[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v39; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 1552);
  v35 = v2;
  v33 = v2 + 832LL * *(unsigned int *)(a1 + 1560);
  if ( v2 == v33 )
  {
    v19 = v2 + 832LL * *(unsigned int *)(a1 + 1560);
    goto LABEL_52;
  }
  v4 = *(_QWORD *)(a1 + 1552);
  do
  {
    while ( *(_QWORD *)(v4 + 824) >= *a2 )
    {
      if ( v4 != v35 )
      {
        v5 = *(_BYTE **)v4;
        v6 = *(_BYTE **)v35;
        if ( *(_QWORD *)v4 != v4 + 16 )
        {
          if ( v6 == (_BYTE *)(v35 + 16) )
          {
            *(_QWORD *)v35 = v5;
            *(_QWORD *)(v35 + 8) = *(_QWORD *)(v4 + 8);
            v7 = *(_QWORD *)(v4 + 16);
            *(_QWORD *)(v35 + 16) = v7;
          }
          else
          {
            *(_QWORD *)v35 = v5;
            v7 = *(_QWORD *)(v35 + 16);
            *(_QWORD *)(v35 + 8) = *(_QWORD *)(v4 + 8);
            *(_QWORD *)(v35 + 16) = *(_QWORD *)(v4 + 16);
            if ( v6 )
            {
              *(_QWORD *)v4 = v6;
              *(_QWORD *)(v4 + 16) = v7;
              goto LABEL_8;
            }
          }
          *(_QWORD *)v4 = v4 + 16;
          v6 = (_BYTE *)(v4 + 16);
LABEL_8:
          *(_QWORD *)(v4 + 8) = 0;
          *v6 = 0;
          *(_BYTE *)(v35 + 32) = *(_BYTE *)(v4 + 32);
          *(_BYTE *)(v35 + 33) = *(_BYTE *)(v4 + 33);
          sub_15ED230(v35 + 40, v4 + 40, v7);
          *(_QWORD *)(v35 + 824) = *(_QWORD *)(v4 + 824);
          v8 = 0x4EC4EC4EC4EC4EC5LL * ((v35 - *(_QWORD *)(a1 + 1552)) >> 6);
          v9 = sub_15EDF70(a1, v35, v37);
          v10 = v37[0];
          if ( v9 )
          {
LABEL_15:
            *(_DWORD *)(v10 + 40) = v8;
            goto LABEL_16;
          }
          v11 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v12 = (v11 >> 1) + 1;
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v13 = 32;
            if ( (unsigned int)(4 * v12) < 0x60 )
              goto LABEL_11;
          }
          else
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( 4 * v12 < 3 * v13 )
            {
LABEL_11:
              if ( v13 - (v12 + *(_DWORD *)(a1 + 12)) > v13 >> 3 )
                goto LABEL_12;
              v30 = a1;
LABEL_65:
              sub_15EF360(v30, v13);
              v31 = v30;
              sub_15EDF70(v30, v35, v37);
              v10 = v37[0];
              v12 = (*(_DWORD *)(v31 + 8) >> 1) + 1;
LABEL_12:
              *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v12);
              if ( *(_WORD *)(v10 + 32) || *(_BYTE *)(v10 + 32) && !*(_BYTE *)(v10 + 33) && *(_QWORD *)(v10 + 8) )
                --*(_DWORD *)(a1 + 12);
              sub_2240AE0(v10, v35);
              *(_BYTE *)(v10 + 32) = *(_BYTE *)(v35 + 32);
              v14 = *(_BYTE *)(v35 + 33);
              *(_DWORD *)(v10 + 40) = 0;
              *(_BYTE *)(v10 + 33) = v14;
              goto LABEL_15;
            }
          }
          v30 = a1;
          v13 *= 2;
          goto LABEL_65;
        }
        v7 = *(_QWORD *)(v4 + 8);
        if ( v7 )
        {
          if ( v7 == 1 )
          {
            *v6 = *(_BYTE *)(v4 + 16);
            v7 = *(_QWORD *)(v4 + 8);
            v32 = *(_BYTE **)v35;
            *(_QWORD *)(v35 + 8) = v7;
            v32[v7] = 0;
            v6 = *(_BYTE **)v4;
            goto LABEL_8;
          }
          memcpy(v6, v5, v7);
          v7 = *(_QWORD *)(v4 + 8);
          v6 = *(_BYTE **)v35;
        }
        *(_QWORD *)(v35 + 8) = v7;
        v6[v7] = 0;
        v6 = *(_BYTE **)v4;
        goto LABEL_8;
      }
LABEL_16:
      v35 += 832;
LABEL_17:
      v4 += 832;
      if ( v33 == v4 )
        goto LABEL_23;
    }
    v15 = sub_15EDF70(a1, v4, v37);
    v16 = v37[0];
    if ( !v15 )
      goto LABEL_17;
    v17 = v37[0];
    v37[0] = v38;
    v37[1] = 0;
    LOBYTE(v38[0]) = 0;
    v39 = 257;
    sub_2240AE0(v17, v37);
    *(_WORD *)(v16 + 32) = v39;
    if ( (_QWORD *)v37[0] != v38 )
      j_j___libc_free_0(v37[0], v38[0] + 1LL);
    v4 += 832;
    v18 = *(_DWORD *)(a1 + 8);
    ++*(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 8) = (2 * (v18 >> 1) - 2) | v18 & 1;
  }
  while ( v33 != v4 );
LABEL_23:
  v19 = *(_QWORD *)(a1 + 1552);
  v33 = v19 + 832LL * *(unsigned int *)(a1 + 1560);
  if ( v35 != v33 )
  {
    do
    {
      v33 -= 832;
      v36 = *(_QWORD *)(v33 + 40);
      v20 = v36 + 192LL * *(unsigned int *)(v33 + 48);
      if ( v36 != v20 )
      {
        do
        {
          v21 = *(unsigned int *)(v20 - 120);
          v22 = *(_QWORD *)(v20 - 128);
          v20 -= 192LL;
          v23 = v22 + 56 * v21;
          if ( v22 != v23 )
          {
            do
            {
              v24 = *(unsigned int *)(v23 - 40);
              v25 = *(_QWORD *)(v23 - 48);
              v23 -= 56LL;
              v24 *= 32;
              v26 = (_QWORD *)(v25 + v24);
              if ( v25 != v25 + v24 )
              {
                do
                {
                  v26 -= 4;
                  if ( (_QWORD *)*v26 != v26 + 2 )
                    j_j___libc_free_0(*v26, v26[2] + 1LL);
                }
                while ( (_QWORD *)v25 != v26 );
                v25 = *(_QWORD *)(v23 + 8);
              }
              if ( v25 != v23 + 24 )
                _libc_free(v25);
            }
            while ( v22 != v23 );
            v22 = *(_QWORD *)(v20 + 64);
          }
          if ( v22 != v20 + 80 )
            _libc_free(v22);
          v27 = *(_QWORD *)(v20 + 16);
          v28 = (_QWORD *)(v27 + 32LL * *(unsigned int *)(v20 + 24));
          if ( (_QWORD *)v27 != v28 )
          {
            do
            {
              v28 -= 4;
              if ( (_QWORD *)*v28 != v28 + 2 )
                j_j___libc_free_0(*v28, v28[2] + 1LL);
            }
            while ( (_QWORD *)v27 != v28 );
            v27 = *(_QWORD *)(v20 + 16);
          }
          if ( v27 != v20 + 32 )
            _libc_free(v27);
        }
        while ( v36 != v20 );
        v36 = *(_QWORD *)(v33 + 40);
      }
      if ( v36 != v33 + 56 )
        _libc_free(v36);
      if ( *(_QWORD *)v33 != v33 + 16 )
        j_j___libc_free_0(*(_QWORD *)v33, *(_QWORD *)(v33 + 16) + 1LL);
    }
    while ( v35 != v33 );
    v19 = *(_QWORD *)(a1 + 1552);
  }
LABEL_52:
  result = 0x4EC4EC4EC4EC4EC5LL * ((v33 - v19) >> 6);
  *(_DWORD *)(a1 + 1560) = result;
  return result;
}
