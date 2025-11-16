// Function: sub_34E00F0
// Address: 0x34e00f0
//
char __fastcall sub_34E00F0(_QWORD *a1, __int64 a2)
{
  int v4; // eax
  __int64 (*v5)(); // rax
  __int64 v6; // rbx
  int v7; // r12d
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdx
  unsigned int v12; // esi
  char *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 i; // rsi
  _QWORD *v17; // rdx
  unsigned int v18; // edx
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdi
  char *v28; // rdx
  unsigned int v29; // esi
  int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // r11
  __int64 v34; // rdx
  __int64 v35; // r10
  unsigned __int64 v36; // rcx
  unsigned int j; // r11d
  _QWORD *v38; // rdx
  __int64 v39; // rdx
  __int16 *v40; // rdx
  __int16 *v41; // r11
  unsigned int v42; // r10d
  unsigned __int64 v43; // rcx
  __int16 *v44; // rsi
  _QWORD *v45; // rdx
  char v47; // [rsp+Fh] [rbp-41h]
  unsigned int v48; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v49[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a2 + 44);
  if ( (v4 & 4) == 0 && (v4 & 8) != 0 )
  {
    LOBYTE(v5) = sub_2E88A90(a2, 128, 1);
    if ( (_BYTE)v5 )
      goto LABEL_4;
  }
  else
  {
    LOBYTE(v5) = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 7;
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x80u) != 0LL )
    {
LABEL_4:
      v47 = 1;
      goto LABEL_5;
    }
  }
  v30 = *(_DWORD *)(a2 + 44);
  if ( (v30 & 4) != 0 || (v30 & 8) == 0 )
  {
    LOBYTE(v5) = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x80000000LL) != 0;
    v47 = (char)v5;
  }
  else
  {
    LOBYTE(v5) = sub_2E88A90(a2, 0x80000000LL, 1);
    v47 = (char)v5;
  }
  if ( v47 )
    goto LABEL_4;
  v31 = a1[3];
  v5 = *(__int64 (**)())(*(_QWORD *)v31 + 920LL);
  if ( v5 != sub_2DB1B30 )
  {
    LOBYTE(v5) = ((__int64 (__fastcall *)(__int64, __int64))v5)(v31, a2);
    v47 = (char)v5;
  }
LABEL_5:
  v6 = 0;
  v7 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( !v7 )
    return (char)v5;
  do
  {
    v8 = *(_QWORD *)(a2 + 32) + 40 * v6;
    if ( *(_BYTE *)v8 )
      goto LABEL_7;
    v9 = *(unsigned int *)(v8 + 8);
    v48 = v9;
    if ( !(_DWORD)v9 )
      goto LABEL_7;
    v10 = *(_QWORD *)(a2 + 16);
    if ( *(unsigned __int16 *)(v10 + 2) > (unsigned int)v6 )
    {
      v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a1[3] + 16LL))(
              a1[3],
              v10,
              (unsigned int)v6,
              a1[4],
              a1[1]);
      v11 = (__int64 *)(a1[15] + 8LL * v48);
      v12 = v48;
      if ( *v11 )
      {
        if ( v25 && *v11 == v25 )
          goto LABEL_13;
      }
      else if ( v25 )
      {
        *v11 = v25;
        v12 = v48;
        goto LABEL_13;
      }
    }
    else
    {
      v11 = (__int64 *)(a1[15] + 8 * v9);
    }
    *v11 = -1;
    v12 = v48;
LABEL_13:
    v13 = sub_E922F0((_QWORD *)a1[4], v12);
    v14 = a1[15];
    for ( i = (__int64)&v13[2 * v15 - 2]; (char *)i != v13; v13 += 2 )
    {
      v17 = (_QWORD *)(v14 + 8LL * *(unsigned __int16 *)v13);
      if ( *v17 )
      {
        *v17 = -1;
        *(_QWORD *)(a1[15] + 8LL * v48) = -1;
        v14 = a1[15];
      }
    }
    if ( *(_QWORD *)(v14 + 8LL * v48) != -1 )
    {
      v49[0] = v8;
      sub_34E0050(a1 + 18, &v48, v49);
    }
    if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
    {
      if ( v47 )
      {
        v18 = v48;
        v19 = a1[30];
        if ( (*(_QWORD *)(v19 + 8LL * (v48 >> 6)) & (1LL << v48)) == 0 )
        {
          v20 = a1[4];
          v21 = *(unsigned int *)(*(_QWORD *)(v20 + 8) + 24LL * v48 + 4);
          v22 = *(_QWORD *)(v20 + 56);
          v23 = v48;
          v24 = v22 + 2 * v21;
          if ( v24 )
          {
            while ( 1 )
            {
              v24 += 2;
              *(_QWORD *)(v19 + ((v23 >> 3) & 0x1FF8)) |= 1LL << v23;
              if ( !*(_WORD *)(v24 - 2) )
                break;
              v18 += *(__int16 *)(v24 - 2);
              v19 = a1[30];
              v23 = v18;
            }
          }
        }
      }
    }
LABEL_7:
    ++v6;
  }
  while ( v7 != (_DWORD)v6 );
  LODWORD(v5) = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( (_DWORD)v5 )
  {
    v26 = 5LL * (unsigned int)v5;
    v5 = 0;
    v27 = 8 * v26;
    do
    {
      v28 = (char *)v5 + *(_QWORD *)(a2 + 32);
      if ( !*v28 )
      {
        v29 = *((_DWORD *)v28 + 2);
        if ( v29 )
        {
          if ( (v28[3] & 0x10) != 0 && (*((_WORD *)v28 + 1) & 0xFF0) != 0 && *(_QWORD *)(a1[15] + 8LL * v29) == -1 )
          {
            v32 = a1[4];
            v33 = *(_QWORD *)(v32 + 56);
            v34 = 24LL * v29 + *(_QWORD *)(v32 + 8);
            v35 = v33 + 2LL * *(unsigned int *)(v34 + 4);
            v36 = v29;
            if ( v35 )
            {
              for ( j = v29; ; v36 = j )
              {
                v35 += 2;
                v38 = (_QWORD *)(a1[30] + ((v36 >> 3) & 0x1FF8));
                *v38 |= 1LL << v36;
                if ( !*(_WORD *)(v35 - 2) )
                  break;
                j += *(__int16 *)(v35 - 2);
              }
              v39 = a1[4];
              v33 = *(_QWORD *)(v39 + 56);
              v34 = *(_QWORD *)(v39 + 8) + 24LL * v29;
            }
            v40 = (__int16 *)(v33 + 2LL * *(unsigned int *)(v34 + 8));
            v41 = v40 + 1;
            LODWORD(v40) = *v40;
            v42 = v29 + (_DWORD)v40;
            if ( (_WORD)v40 )
            {
              v43 = v42;
              v44 = v41;
              while ( 1 )
              {
                ++v44;
                v45 = (_QWORD *)(a1[30] + ((v43 >> 3) & 0x1FF8));
                *v45 |= 1LL << v43;
                if ( !*(v44 - 1) )
                  break;
                v42 += *(v44 - 1);
                v43 = v42;
              }
            }
          }
        }
      }
      v5 = (__int64 (*)())((char *)v5 + 40);
    }
    while ( (__int64 (*)())v27 != v5 );
  }
  return (char)v5;
}
