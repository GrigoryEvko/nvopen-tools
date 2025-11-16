// Function: sub_35400A0
// Address: 0x35400a0
//
__int64 __fastcall sub_35400A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rdx
  char v11; // dl
  __int64 v12; // r15
  unsigned int v13; // ecx
  unsigned int v14; // edx
  bool v15; // si
  unsigned int v16; // edx
  unsigned int v17; // ecx
  int v18; // esi
  int v19; // ecx
  bool v20; // dl
  __int64 v21; // rcx
  __int64 v22; // rdx
  char v23; // dl
  __int64 v24; // r12
  __int64 v25; // r14
  __int64 v26; // r13
  char v27; // dl
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // r14
  __int64 v32; // r13
  __int64 v33; // rbx
  __int64 v34; // r12
  char v35; // al
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v39; // [rsp+0h] [rbp-40h]

  v5 = a1;
  v7 = a3;
  if ( a4 != a3 && a2 != a1 )
  {
    v8 = a5 + 32;
    while ( 1 )
    {
      v13 = *(_DWORD *)(v7 + 52);
      v14 = *(_DWORD *)(v5 + 52);
      v15 = v13 > v14;
      if ( v13 != v14
        || (v16 = *(_DWORD *)(v7 + 64)) != 0 && (v17 = *(_DWORD *)(v5 + 64), v16 != v17) && (v15 = v16 < v17, v17) )
      {
        if ( !v15 )
          goto LABEL_17;
      }
      else
      {
        v18 = *(_DWORD *)(v7 + 56);
        v19 = *(_DWORD *)(v5 + 56);
        v20 = v18 < v19;
        if ( v18 == v19 )
          v20 = *(_DWORD *)(v7 + 60) > *(_DWORD *)(v5 + 60);
        if ( !v20 )
        {
LABEL_17:
          sub_C7D6A0(*(_QWORD *)(v8 - 24), 8LL * *(unsigned int *)(v8 - 8), 8);
          ++*(_QWORD *)(v8 - 32);
          *(_DWORD *)(v8 - 8) = 0;
          *(_QWORD *)(v8 - 24) = 0;
          *(_DWORD *)(v8 - 16) = 0;
          *(_DWORD *)(v8 - 12) = 0;
          v21 = *(_QWORD *)(v5 + 8);
          ++*(_QWORD *)v5;
          v22 = *(_QWORD *)(v8 - 24);
          *(_QWORD *)(v8 - 24) = v21;
          LODWORD(v21) = *(_DWORD *)(v5 + 16);
          *(_QWORD *)(v5 + 8) = v22;
          LODWORD(v22) = *(_DWORD *)(v8 - 16);
          *(_DWORD *)(v8 - 16) = v21;
          LODWORD(v21) = *(_DWORD *)(v5 + 20);
          *(_DWORD *)(v5 + 16) = v22;
          LODWORD(v22) = *(_DWORD *)(v8 - 12);
          *(_DWORD *)(v8 - 12) = v21;
          LODWORD(v21) = *(_DWORD *)(v5 + 24);
          *(_DWORD *)(v5 + 20) = v22;
          LODWORD(v22) = *(_DWORD *)(v8 - 8);
          *(_DWORD *)(v8 - 8) = v21;
          *(_DWORD *)(v5 + 24) = v22;
          if ( v8 != v5 + 32 )
          {
            if ( *(_DWORD *)(v5 + 40) )
            {
              if ( *(_QWORD *)v8 != v8 + 16 )
                _libc_free(*(_QWORD *)v8);
              *(_QWORD *)v8 = *(_QWORD *)(v5 + 32);
              *(_DWORD *)(v8 + 8) = *(_DWORD *)(v5 + 40);
              *(_DWORD *)(v8 + 12) = *(_DWORD *)(v5 + 44);
              *(_QWORD *)(v5 + 32) = v5 + 48;
              *(_QWORD *)(v5 + 40) = 0;
            }
            else
            {
              *(_DWORD *)(v8 + 8) = 0;
            }
          }
          v23 = *(_BYTE *)(v5 + 48);
          v5 += 88;
          v12 = v8 + 56;
          v8 += 88;
          *(_BYTE *)(v8 - 72) = v23;
          *(_DWORD *)(v8 - 68) = *(_DWORD *)(v5 - 36);
          *(_DWORD *)(v8 - 64) = *(_DWORD *)(v5 - 32);
          *(_DWORD *)(v8 - 60) = *(_DWORD *)(v5 - 28);
          *(_DWORD *)(v8 - 56) = *(_DWORD *)(v5 - 24);
          *(_QWORD *)(v8 - 48) = *(_QWORD *)(v5 - 16);
          *(_DWORD *)(v8 - 40) = *(_DWORD *)(v5 - 8);
          if ( a2 == v5 )
            goto LABEL_21;
          goto LABEL_9;
        }
      }
      sub_C7D6A0(*(_QWORD *)(v8 - 24), 8LL * *(unsigned int *)(v8 - 8), 8);
      ++*(_QWORD *)(v8 - 32);
      *(_DWORD *)(v8 - 8) = 0;
      *(_QWORD *)(v8 - 24) = 0;
      *(_DWORD *)(v8 - 16) = 0;
      *(_DWORD *)(v8 - 12) = 0;
      v9 = *(_QWORD *)(v7 + 8);
      ++*(_QWORD *)v7;
      v10 = *(_QWORD *)(v8 - 24);
      *(_QWORD *)(v8 - 24) = v9;
      LODWORD(v9) = *(_DWORD *)(v7 + 16);
      *(_QWORD *)(v7 + 8) = v10;
      LODWORD(v10) = *(_DWORD *)(v8 - 16);
      *(_DWORD *)(v8 - 16) = v9;
      LODWORD(v9) = *(_DWORD *)(v7 + 20);
      *(_DWORD *)(v7 + 16) = v10;
      LODWORD(v10) = *(_DWORD *)(v8 - 12);
      *(_DWORD *)(v8 - 12) = v9;
      LODWORD(v9) = *(_DWORD *)(v7 + 24);
      *(_DWORD *)(v7 + 20) = v10;
      LODWORD(v10) = *(_DWORD *)(v8 - 8);
      *(_DWORD *)(v8 - 8) = v9;
      *(_DWORD *)(v7 + 24) = v10;
      if ( v8 != v7 + 32 )
      {
        if ( *(_DWORD *)(v7 + 40) )
        {
          if ( *(_QWORD *)v8 != v8 + 16 )
            _libc_free(*(_QWORD *)v8);
          *(_QWORD *)v8 = *(_QWORD *)(v7 + 32);
          *(_DWORD *)(v8 + 8) = *(_DWORD *)(v7 + 40);
          *(_DWORD *)(v8 + 12) = *(_DWORD *)(v7 + 44);
          *(_QWORD *)(v7 + 32) = v7 + 48;
          *(_QWORD *)(v7 + 40) = 0;
        }
        else
        {
          *(_DWORD *)(v8 + 8) = 0;
        }
      }
      v11 = *(_BYTE *)(v7 + 48);
      v12 = v8 + 56;
      v7 += 88;
      v8 += 88;
      *(_BYTE *)(v8 - 72) = v11;
      *(_DWORD *)(v8 - 68) = *(_DWORD *)(v7 - 36);
      *(_DWORD *)(v8 - 64) = *(_DWORD *)(v7 - 32);
      *(_DWORD *)(v8 - 60) = *(_DWORD *)(v7 - 28);
      *(_DWORD *)(v8 - 56) = *(_DWORD *)(v7 - 24);
      *(_QWORD *)(v8 - 48) = *(_QWORD *)(v7 - 16);
      *(_DWORD *)(v8 - 40) = *(_DWORD *)(v7 - 8);
      if ( a2 == v5 )
        goto LABEL_21;
LABEL_9:
      if ( a4 == v7 )
        goto LABEL_21;
    }
  }
  v12 = a5;
LABEL_21:
  v39 = a2 - v5;
  v24 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - v5) >> 3);
  if ( a2 - v5 > 0 )
  {
    v25 = v5 + 32;
    v26 = v12 + 32;
    do
    {
      sub_C7D6A0(*(_QWORD *)(v26 - 24), 8LL * *(unsigned int *)(v26 - 8), 8);
      *(_DWORD *)(v26 - 8) = 0;
      *(_QWORD *)(v26 - 24) = 0;
      *(_DWORD *)(v26 - 16) = 0;
      *(_DWORD *)(v26 - 12) = 0;
      ++*(_QWORD *)(v26 - 32);
      v28 = *(_QWORD *)(v25 - 24);
      ++*(_QWORD *)(v25 - 32);
      v29 = *(_QWORD *)(v26 - 24);
      *(_QWORD *)(v26 - 24) = v28;
      LODWORD(v28) = *(_DWORD *)(v25 - 16);
      *(_QWORD *)(v25 - 24) = v29;
      LODWORD(v29) = *(_DWORD *)(v26 - 16);
      *(_DWORD *)(v26 - 16) = v28;
      LODWORD(v28) = *(_DWORD *)(v25 - 12);
      *(_DWORD *)(v25 - 16) = v29;
      LODWORD(v29) = *(_DWORD *)(v26 - 12);
      *(_DWORD *)(v26 - 12) = v28;
      LODWORD(v28) = *(_DWORD *)(v25 - 8);
      *(_DWORD *)(v25 - 12) = v29;
      LODWORD(v29) = *(_DWORD *)(v26 - 8);
      *(_DWORD *)(v26 - 8) = v28;
      *(_DWORD *)(v25 - 8) = v29;
      if ( v26 != v25 )
      {
        if ( *(_DWORD *)(v25 + 8) )
        {
          if ( *(_QWORD *)v26 != v26 + 16 )
            _libc_free(*(_QWORD *)v26);
          *(_QWORD *)v26 = *(_QWORD *)v25;
          *(_DWORD *)(v26 + 8) = *(_DWORD *)(v25 + 8);
          *(_DWORD *)(v26 + 12) = *(_DWORD *)(v25 + 12);
          *(_QWORD *)v25 = v25 + 16;
          *(_DWORD *)(v25 + 12) = 0;
          *(_DWORD *)(v25 + 8) = 0;
        }
        else
        {
          *(_DWORD *)(v26 + 8) = 0;
        }
      }
      v27 = *(_BYTE *)(v25 + 16);
      v26 += 88;
      v25 += 88;
      *(_BYTE *)(v26 - 72) = v27;
      *(_DWORD *)(v26 - 68) = *(_DWORD *)(v25 - 68);
      *(_DWORD *)(v26 - 64) = *(_DWORD *)(v25 - 64);
      *(_DWORD *)(v26 - 60) = *(_DWORD *)(v25 - 60);
      *(_DWORD *)(v26 - 56) = *(_DWORD *)(v25 - 56);
      *(_QWORD *)(v26 - 48) = *(_QWORD *)(v25 - 48);
      *(_DWORD *)(v26 - 40) = *(_DWORD *)(v25 - 40);
      --v24;
    }
    while ( v24 );
    v30 = v39;
    if ( v39 <= 0 )
      v30 = 88;
    v12 += v30;
  }
  v31 = a4 - v7;
  v32 = 0x2E8BA2E8BA2E8BA3LL * ((a4 - v7) >> 3);
  if ( a4 - v7 > 0 )
  {
    v33 = v7 + 32;
    v34 = v12 + 32;
    do
    {
      sub_C7D6A0(*(_QWORD *)(v34 - 24), 8LL * *(unsigned int *)(v34 - 8), 8);
      ++*(_QWORD *)(v34 - 32);
      *(_DWORD *)(v34 - 8) = 0;
      *(_QWORD *)(v34 - 24) = 0;
      *(_DWORD *)(v34 - 16) = 0;
      *(_DWORD *)(v34 - 12) = 0;
      v36 = *(_QWORD *)(v33 - 24);
      ++*(_QWORD *)(v33 - 32);
      v37 = *(_QWORD *)(v34 - 24);
      *(_QWORD *)(v34 - 24) = v36;
      LODWORD(v36) = *(_DWORD *)(v33 - 16);
      *(_QWORD *)(v33 - 24) = v37;
      LODWORD(v37) = *(_DWORD *)(v34 - 16);
      *(_DWORD *)(v34 - 16) = v36;
      LODWORD(v36) = *(_DWORD *)(v33 - 12);
      *(_DWORD *)(v33 - 16) = v37;
      LODWORD(v37) = *(_DWORD *)(v34 - 12);
      *(_DWORD *)(v34 - 12) = v36;
      LODWORD(v36) = *(_DWORD *)(v33 - 8);
      *(_DWORD *)(v33 - 12) = v37;
      LODWORD(v37) = *(_DWORD *)(v34 - 8);
      *(_DWORD *)(v34 - 8) = v36;
      *(_DWORD *)(v33 - 8) = v37;
      if ( v34 != v33 )
      {
        if ( *(_DWORD *)(v33 + 8) )
        {
          if ( *(_QWORD *)v34 != v34 + 16 )
            _libc_free(*(_QWORD *)v34);
          *(_QWORD *)v34 = *(_QWORD *)v33;
          *(_DWORD *)(v34 + 8) = *(_DWORD *)(v33 + 8);
          *(_DWORD *)(v34 + 12) = *(_DWORD *)(v33 + 12);
          *(_QWORD *)v33 = v33 + 16;
          *(_DWORD *)(v33 + 12) = 0;
          *(_DWORD *)(v33 + 8) = 0;
        }
        else
        {
          *(_DWORD *)(v34 + 8) = 0;
        }
      }
      v35 = *(_BYTE *)(v33 + 16);
      v34 += 88;
      v33 += 88;
      *(_BYTE *)(v34 - 72) = v35;
      *(_DWORD *)(v34 - 68) = *(_DWORD *)(v33 - 68);
      *(_DWORD *)(v34 - 64) = *(_DWORD *)(v33 - 64);
      *(_DWORD *)(v34 - 60) = *(_DWORD *)(v33 - 60);
      *(_DWORD *)(v34 - 56) = *(_DWORD *)(v33 - 56);
      *(_QWORD *)(v34 - 48) = *(_QWORD *)(v33 - 48);
      *(_DWORD *)(v34 - 40) = *(_DWORD *)(v33 - 40);
      --v32;
    }
    while ( v32 );
    if ( v31 <= 0 )
      v31 = 88;
    v12 += v31;
  }
  return v12;
}
