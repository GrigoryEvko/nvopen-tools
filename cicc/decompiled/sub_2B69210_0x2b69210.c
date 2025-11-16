// Function: sub_2B69210
// Address: 0x2b69210
//
__int64 __fastcall sub_2B69210(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 *v9; // rax
  char v11; // dl
  __int64 v12; // rax
  unsigned int v13; // r13d
  __int64 v14; // rdx
  __int64 v15; // rax
  _BYTE *v16; // rdi
  __int64 *v17; // r15
  unsigned int v18; // eax
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  int v23; // r14d
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)a1;
  if ( *(_BYTE *)(v8 + 28) )
  {
    v9 = *(__int64 **)(v8 + 8);
    a4 = *(unsigned int *)(v8 + 20);
    a3 = &v9[a4];
    if ( v9 != a3 )
    {
      while ( a2 != *v9 )
      {
        if ( a3 == ++v9 )
          goto LABEL_7;
      }
      return 0;
    }
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(v8 + 16) )
    {
      *(_DWORD *)(v8 + 20) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)v8;
      v12 = **(_QWORD **)(a1 + 8);
      if ( v12 == a2 )
        return 1;
LABEL_10:
      if ( *(_DWORD *)(a2 + 152) )
        return 1;
      v13 = *(_DWORD *)(a2 + 120);
      if ( v13 )
        return 1;
      v14 = *(_QWORD *)(a2 + 184);
      if ( v12 == v14 )
      {
        if ( v14 )
          return 1;
      }
      v15 = *(_QWORD *)(v12 + 184);
      if ( a2 == v15 )
      {
        if ( v15 )
          return 1;
      }
      if ( **(_BYTE **)(a1 + 16) && v14 && !*(_DWORD *)(v14 + 200) )
        return 1;
      v16 = *(_BYTE **)(a2 + 416);
      v17 = *(__int64 **)(a1 + 24);
      if ( *v16 != 85
        || (v18 = sub_9B78C0((__int64)v16, *(__int64 **)(v17[1] + 3304)),
            v19 = *(_QWORD *)(a2 + 416),
            v13 = v18,
            *(_BYTE *)v19 != 85) )
      {
        LODWORD(v27) = *(_DWORD *)(a2 + 248);
LABEL_31:
        v33 = (unsigned int)v27;
        if ( (_DWORD)v27 )
        {
          v28 = 0;
          while ( 1 )
          {
            if ( !v13 || !sub_9B75A0(v13, v28, *v17) )
            {
              v29 = sub_2B68AE0(v17[1], a2, v28);
              v30 = v29;
              if ( *(_DWORD *)(v29 + 104) == 3 )
              {
                v31 = *(_QWORD *)(v29 + 416);
                if ( v31 )
                {
                  if ( *(_QWORD *)(v29 + 424) )
                  {
                    v32 = sub_2B31EF0(v17[1], v31, *(char **)v29, *(unsigned int *)(v29 + 8), 0);
                    if ( v32 )
                      v30 = v32;
                  }
                }
              }
              if ( !*(_DWORD *)(v30 + 152) && !*(_DWORD *)(v30 + 120) )
                break;
            }
            if ( v33 == ++v28 )
              return 1;
          }
          return 0;
        }
        return 1;
      }
      if ( *(char *)(v19 + 7) < 0 )
      {
        v20 = sub_BD2BC0(*(_QWORD *)(a2 + 416));
        v22 = v20 + v21;
        if ( *(char *)(v19 + 7) >= 0 )
        {
          if ( (unsigned int)(v22 >> 4) )
            goto LABEL_49;
        }
        else if ( (unsigned int)((v22 - sub_BD2BC0(v19)) >> 4) )
        {
          if ( *(char *)(v19 + 7) < 0 )
          {
            v23 = *(_DWORD *)(sub_BD2BC0(v19) + 8);
            if ( *(char *)(v19 + 7) >= 0 )
              BUG();
            v24 = sub_BD2BC0(v19);
            v26 = 32LL * (unsigned int)(*(_DWORD *)(v24 + v25 - 4) - v23);
            goto LABEL_30;
          }
LABEL_49:
          BUG();
        }
      }
      v26 = 0;
LABEL_30:
      v27 = (32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) - 32 - v26) >> 5;
      goto LABEL_31;
    }
  }
  sub_C8CC70(v8, a2, (__int64)a3, a4, a5, a6);
  if ( !v11 )
    return 0;
  v12 = **(_QWORD **)(a1 + 8);
  if ( v12 != a2 )
    goto LABEL_10;
  return 1;
}
