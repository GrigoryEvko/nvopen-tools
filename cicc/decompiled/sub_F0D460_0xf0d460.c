// Function: sub_F0D460
// Address: 0xf0d460
//
__int64 __fastcall sub_F0D460(__int64 a1, __int64 a2, unsigned __int8 *a3, char *a4)
{
  int v5; // eax
  unsigned __int8 v6; // si
  unsigned int v8; // eax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 *v12; // r13
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned __int8 *v18; // rdx
  char *v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rsi
  unsigned int v26; // esi
  int v27; // ebx
  unsigned int v28; // edi
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // r10
  __int64 v32; // rcx
  __int64 v33; // rax
  unsigned int v34; // eax
  unsigned __int8 *v36; // [rsp+8h] [rbp-28h]

  v5 = *a3;
  if ( (unsigned __int8)v5 <= 0x1Cu )
    goto LABEL_4;
  v6 = *a4;
  if ( (unsigned __int8)*a4 <= 0x1Cu || (_BYTE)v5 != v6 )
    goto LABEL_4;
  if ( v5 == 85 )
  {
    v22 = *((_QWORD *)a3 - 4);
    if ( v22 && !*(_BYTE *)v22 && *(_QWORD *)(v22 + 24) == *((_QWORD *)a3 + 10) && (*(_BYTE *)(v22 + 33) & 0x20) != 0 )
    {
      v34 = *(_DWORD *)(v22 + 36);
      if ( v34 > 0x14A )
      {
        if ( v34 - 365 >= 2 )
          a3 = 0;
      }
      else if ( v34 <= 0x148 )
      {
        a3 = 0;
      }
    }
    else
    {
      a3 = 0;
    }
    if ( v6 != 85 )
      goto LABEL_4;
    v23 = *((_QWORD *)a4 - 4);
    if ( !v23 || *(_BYTE *)v23 || *(_QWORD *)(v23 + 24) != *((_QWORD *)a4 + 10) || (*(_BYTE *)(v23 + 33) & 0x20) == 0 )
      goto LABEL_4;
    v24 = *(_DWORD *)(v23 + 36);
    if ( v24 > 0x14A )
    {
      if ( v24 - 365 > 1 )
        goto LABEL_4;
    }
    else if ( v24 <= 0x148 )
    {
      goto LABEL_4;
    }
    if ( !a3 )
      goto LABEL_4;
    v25 = *((_QWORD *)a3 - 4);
    if ( !v25 || *(_BYTE *)v25 || *(_QWORD *)(v25 + 24) != *((_QWORD *)a3 + 10) )
      BUG();
    v26 = *(_DWORD *)(v25 + 36);
    if ( v26 == 365 )
    {
      v27 = 34;
    }
    else if ( v26 > 0x16D )
    {
      if ( v26 != 366 )
        goto LABEL_83;
      v27 = 36;
    }
    else if ( v26 == 329 )
    {
      v27 = 38;
    }
    else
    {
      if ( v26 != 330 )
        goto LABEL_83;
      v27 = 40;
    }
    if ( v24 == 365 )
    {
      v28 = 34;
      goto LABEL_53;
    }
    if ( v24 > 0x16D )
    {
      if ( v24 == 366 )
      {
        v28 = 36;
        goto LABEL_53;
      }
    }
    else
    {
      if ( v24 == 329 )
      {
        v28 = 38;
LABEL_53:
        v36 = a3;
        if ( (unsigned int)sub_B52F50(v28) == v27 )
        {
          if ( (v29 = *((_DWORD *)v36 + 1) & 0x7FFFFFF,
                v30 = *(_QWORD *)&v36[-32 * v29],
                v31 = *(_QWORD *)&a4[-32 * (*((_DWORD *)a4 + 1) & 0x7FFFFFF)],
                v32 = *(_QWORD *)&a4[32 * (1LL - (*((_DWORD *)a4 + 1) & 0x7FFFFFF))],
                v31 == v30)
            && *(_QWORD *)&v36[32 * (1 - v29)] == v32
            || v30 == v32 && v31 == *(_QWORD *)&v36[32 * (1 - v29)] )
          {
            *(_QWORD *)a1 = v30;
            v33 = *(_QWORD *)&v36[32 * (1 - v29)];
            *(_BYTE *)(a1 + 16) = 1;
            *(_QWORD *)(a1 + 8) = v33;
            return a1;
          }
        }
        goto LABEL_4;
      }
      if ( v24 == 330 )
      {
        v28 = 40;
        goto LABEL_53;
      }
    }
LABEL_83:
    BUG();
  }
  if ( v5 == 86 )
  {
    if ( (a3[7] & 0x40) != 0 )
      v18 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
    else
      v18 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    if ( (a4[7] & 0x40) != 0 )
      v19 = (char *)*((_QWORD *)a4 - 1);
    else
      v19 = &a4[-32 * (*((_DWORD *)a4 + 1) & 0x7FFFFFF)];
    if ( *(_QWORD *)v19 == *(_QWORD *)v18 )
    {
      v20 = *((_QWORD *)v18 + 4);
      if ( v20 == *((_QWORD *)v19 + 8) )
      {
        v21 = *((_QWORD *)v18 + 8);
        if ( v21 == *((_QWORD *)v19 + 4) )
        {
          *(_QWORD *)a1 = v20;
          *(_QWORD *)(a1 + 8) = v21;
          *(_BYTE *)(a1 + 16) = 1;
          return a1;
        }
      }
    }
  }
  else if ( v5 == 84 && *((_QWORD *)a3 + 5) == *((_QWORD *)a4 + 5) )
  {
    v8 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
    if ( v8 > 1 )
    {
      v9 = v8;
      v10 = *((_DWORD *)a4 + 1) & 0x7FFFFFF;
      if ( v10 == v9 )
      {
        v11 = (__int64 *)*((_QWORD *)a3 - 1);
        v12 = (__int64 *)*((_QWORD *)a4 - 1);
        if ( !memcmp(&v11[4 * *((unsigned int *)a3 + 18)], &v12[4 * *((unsigned int *)a4 + 18)], 8 * v10) )
        {
          v13 = *v11;
          v14 = *v12;
          v15 = 4;
          while ( 1 )
          {
            v16 = v12[v15];
            v17 = v11[v15];
            if ( (v14 != v16 || v13 != v17) && (v14 != v17 || v13 != v16) )
              break;
            v15 += 4;
            if ( 4 * v9 == v15 )
            {
              *(_QWORD *)a1 = v13;
              *(_QWORD *)(a1 + 8) = v14;
              *(_BYTE *)(a1 + 16) = 1;
              return a1;
            }
          }
        }
      }
    }
  }
LABEL_4:
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
