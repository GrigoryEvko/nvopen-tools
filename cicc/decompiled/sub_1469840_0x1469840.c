// Function: sub_1469840
// Address: 0x1469840
//
void __fastcall sub_1469840(__int64 a1, unsigned int a2)
{
  char v3; // bl
  unsigned int v4; // eax
  int v5; // r14d
  __int64 v6; // r12
  __int64 *v7; // rbx
  _BYTE *v8; // r13
  __int64 v9; // rdx
  _BYTE *v10; // rdi
  _BYTE *v11; // rsi
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rax
  bool v14; // zf
  _QWORD *v15; // rdx
  _QWORD *v16; // r12
  _BYTE *v17; // rbx
  _QWORD *v18; // r10
  int v19; // edi
  int v20; // r12d
  _QWORD *v21; // r11
  unsigned int v22; // edx
  _QWORD *v23; // rsi
  __int64 v24; // rcx
  _QWORD *v25; // rdi
  _QWORD *v26; // rsi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rax
  int v29; // edx
  __int64 *v30; // r13
  unsigned int v31; // ebx
  __int64 *v32; // r12
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *i; // rdx
  unsigned __int64 *v36; // rbx
  __int64 v37; // r9
  int v38; // edi
  int v39; // r11d
  _QWORD *v40; // r10
  unsigned int v41; // edx
  _QWORD *v42; // rsi
  __int64 v43; // rcx
  _QWORD *v44; // rdi
  _QWORD *v45; // rsi
  unsigned __int64 v46; // rdi
  int v47; // edx
  __int64 v48; // rax
  _QWORD *v49; // [rsp+10h] [rbp-1E0h]
  __int64 *v50; // [rsp+18h] [rbp-1D8h]
  _BYTE v51[464]; // [rsp+20h] [rbp-1D0h] BYREF

  v3 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v3 )
      return;
    v30 = *(__int64 **)(a1 + 16);
    v31 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
  }
  else
  {
    v4 = sub_1454B60(a2 - 1);
    v5 = v4;
    if ( v4 > 0x40 )
    {
      v6 = 13LL * v4;
      if ( v3 )
        goto LABEL_5;
      v30 = *(__int64 **)(a1 + 16);
      v31 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( v3 )
      {
        v6 = 832;
        v5 = 64;
LABEL_5:
        v7 = (__int64 *)(a1 + 16);
        v50 = (__int64 *)(a1 + 432);
        v49 = (_QWORD *)(a1 + 16);
        v8 = v51;
        do
        {
          v9 = *v7;
          if ( *v7 != -4 && v9 != -16 )
          {
            if ( v8 )
              *(_QWORD *)v8 = v9;
            v10 = v8 + 32;
            v11 = v8 + 72;
            v8 += 104;
            *((_QWORD *)v8 - 12) = v7[1];
            *((_QWORD *)v8 - 11) = v7[2];
            *(v8 - 80) = *((_BYTE *)v7 + 24);
            sub_16CCEE0(v10, v11, 4, v7 + 4);
            v12 = v7[6];
            if ( v7[5] != v12 )
              _libc_free(v12);
          }
          v7 += 13;
        }
        while ( v7 != v50 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v13 = (_QWORD *)sub_22077B0(v6 * 8);
        v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = v13;
        v15 = v13;
        *(_DWORD *)(a1 + 24) = v5;
        if ( !v14 )
        {
          v13 = v49;
          v6 = 52;
          v15 = v49;
        }
        v16 = &v13[v6];
        while ( 1 )
        {
          if ( v15 )
            *v13 = -4;
          v13 += 13;
          if ( v16 == v13 )
            break;
          v15 = v13;
        }
        v17 = v51;
        if ( v8 != v51 )
        {
          while ( 1 )
          {
            v28 = *(_QWORD *)v17;
            if ( *(_QWORD *)v17 != -16 && v28 != -4 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v18 = v49;
                v19 = 3;
              }
              else
              {
                v29 = *(_DWORD *)(a1 + 24);
                v18 = *(_QWORD **)(a1 + 16);
                if ( !v29 )
                  goto LABEL_77;
                v19 = v29 - 1;
              }
              v20 = 1;
              v21 = 0;
              v22 = v19 & (v28 ^ (v28 >> 9));
              v23 = &v18[13 * v22];
              v24 = *v23;
              if ( v28 != *v23 )
              {
                while ( v24 != -4 )
                {
                  if ( v24 == -16 && !v21 )
                    v21 = v23;
                  v22 = v19 & (v20 + v22);
                  v23 = &v18[13 * v22];
                  v24 = *v23;
                  if ( v28 == *v23 )
                    goto LABEL_24;
                  ++v20;
                }
                if ( v21 )
                  v23 = v21;
              }
LABEL_24:
              v25 = v23 + 4;
              v26 = v23 + 9;
              *(v26 - 9) = *(_QWORD *)v17;
              *(v26 - 8) = *((_QWORD *)v17 + 1);
              *(v26 - 7) = *((_QWORD *)v17 + 2);
              *((_BYTE *)v26 - 48) = v17[24];
              sub_16CCEE0(v25, v26, 4, v17 + 32);
              v27 = *((_QWORD *)v17 + 6);
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              if ( v27 != *((_QWORD *)v17 + 5) )
                _libc_free(v27);
            }
            v17 += 104;
            if ( v8 == v17 )
              return;
          }
        }
        return;
      }
      v30 = *(__int64 **)(a1 + 16);
      v31 = *(_DWORD *)(a1 + 24);
      v6 = 832;
      v5 = 64;
    }
    v48 = sub_22077B0(v6 * 8);
    *(_DWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 16) = v48;
  }
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v32 = &v30[13 * v31];
  if ( v14 )
  {
    v33 = *(_QWORD **)(a1 + 16);
    v34 = 13LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = (_QWORD *)(a1 + 16);
    v34 = 52;
  }
  for ( i = &v33[v34]; i != v33; v33 += 13 )
  {
    if ( v33 )
      *v33 = -4;
  }
  v36 = (unsigned __int64 *)v30;
  if ( v32 != v30 )
  {
    do
    {
      v28 = *v36;
      if ( *v36 != -16 && v28 != -4 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v37 = a1 + 16;
          v38 = 3;
        }
        else
        {
          v47 = *(_DWORD *)(a1 + 24);
          v37 = *(_QWORD *)(a1 + 16);
          if ( !v47 )
          {
LABEL_77:
            MEMORY[0] = v28;
            BUG();
          }
          v38 = v47 - 1;
        }
        v39 = 1;
        v40 = 0;
        v41 = v38 & (v28 ^ (v28 >> 9));
        v42 = (_QWORD *)(v37 + 104LL * v41);
        v43 = *v42;
        if ( *v42 != v28 )
        {
          while ( v43 != -4 )
          {
            if ( !v40 && v43 == -16 )
              v40 = v42;
            v41 = v38 & (v39 + v41);
            v42 = (_QWORD *)(v37 + 104LL * v41);
            v43 = *v42;
            if ( v28 == *v42 )
              goto LABEL_45;
            ++v39;
          }
          if ( v40 )
            v42 = v40;
        }
LABEL_45:
        v44 = v42 + 4;
        v45 = v42 + 9;
        *(v45 - 9) = *v36;
        *(v45 - 8) = v36[1];
        *(v45 - 7) = v36[2];
        *((_BYTE *)v45 - 48) = *((_BYTE *)v36 + 24);
        sub_16CCEE0(v44, v45, 4, v36 + 4);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v46 = v36[6];
        if ( v46 != v36[5] )
          _libc_free(v46);
      }
      v36 += 13;
    }
    while ( v32 != (__int64 *)v36 );
  }
  j___libc_free_0(v30);
}
