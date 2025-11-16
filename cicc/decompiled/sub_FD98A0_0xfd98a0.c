// Function: sub_FD98A0
// Address: 0xfd98a0
//
char __fastcall sub_FD98A0(__int64 **a1, unsigned __int8 *a2, __m128i a3)
{
  int v4; // eax
  __m128i *v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // eax
  int v9; // eax
  unsigned __int8 v10; // dl
  int v11; // eax
  int v12; // edx
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r15
  int v18; // r15d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // r15d
  unsigned __int8 *v23; // r14
  unsigned __int8 *v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rcx
  int v29; // eax
  char v30; // si
  __int64 v31; // rdx
  __int64 v32; // rdx
  char v34; // [rsp+Fh] [rbp-61h]
  __m128i v35; // [rsp+10h] [rbp-60h] BYREF

  v4 = *a2;
  if ( (_BYTE)v4 != 61 )
  {
    switch ( (_BYTE)v4 )
    {
      case '>':
        LOBYTE(v5) = sub_FD9730((__int64)a1, (__int64)a2, a3);
        return (char)v5;
      case 'Y':
        LOBYTE(v5) = sub_FD97A0((__int64)a1, (__int64)a2, a3);
        return (char)v5;
      case 'U':
        v31 = *((_QWORD *)a2 - 4);
        if ( v31 )
        {
          if ( !*(_BYTE *)v31
            && *(_QWORD *)(v31 + 24) == *((_QWORD *)a2 + 10)
            && (*(_BYTE *)(v31 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v31 + 36) - 243) <= 2 )
          {
            LOBYTE(v5) = sub_FD97E0((__int64)a1, (__int64)a2, a3);
            return (char)v5;
          }
          if ( !*(_BYTE *)v31
            && *(_QWORD *)(v31 + 24) == *((_QWORD *)a2 + 10)
            && (*(_BYTE *)(v31 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v31 + 36) - 238) <= 4 )
          {
            LOBYTE(v5) = sub_FD9820((__int64)a1, (__int64)a2, a3);
            return (char)v5;
          }
        }
        break;
      default:
        v6 = (unsigned int)(v4 - 34);
        if ( (unsigned __int8)v6 > 0x33u )
          goto LABEL_30;
        v7 = 0x8000000000041LL;
        if ( !_bittest64(&v7, v6) )
          goto LABEL_30;
        break;
    }
    if ( sub_B49E80((__int64)a2) )
    {
      v8 = sub_CF5230(**a1, (__int64)a2, (__int64)(*a1 + 1));
      v9 = (v8 >> 6) | (v8 >> 4) | v8 | (v8 >> 2);
      v10 = *a2;
      v34 = v9 & 3;
      if ( *((_QWORD *)a2 + 2) || v10 != 85 )
      {
        v11 = v10;
        v12 = v10 - 29;
        if ( v11 == 40 )
        {
          v13 = sub_B491D0((__int64)a2);
          goto LABEL_15;
        }
        if ( v12 != 56 )
        {
          v13 = 2;
          if ( v12 != 5 )
            goto LABEL_57;
LABEL_15:
          v14 = -32 - 32LL * v13;
          if ( (a2[7] & 0x80u) != 0 )
          {
            v15 = sub_BD2BC0((__int64)a2);
            v17 = v15 + v16;
            if ( (a2[7] & 0x80u) == 0 )
            {
              if ( (unsigned int)(v17 >> 4) )
                goto LABEL_57;
            }
            else if ( (unsigned int)((v17 - sub_BD2BC0((__int64)a2)) >> 4) )
            {
              if ( (a2[7] & 0x80u) != 0 )
              {
                v18 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
                if ( (a2[7] & 0x80u) == 0 )
                  BUG();
                v19 = sub_BD2BC0((__int64)a2);
                v21 = (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v18);
LABEL_21:
                v22 = 0;
                v23 = &a2[v14 - 32 * v21];
                v24 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
                v5 = &v35;
                if ( v24 == v23 )
                  return (char)v5;
                while ( 1 )
                {
                  v5 = *(__m128i **)(*(_QWORD *)v24 + 8LL);
                  if ( v5->m128i_i8[8] != 14 )
                    goto LABEL_23;
                  sub_D669C0(&v35, (__int64)a2, v22, 0);
                  LOBYTE(v5) = v34 & sub_CF51C0(**a1, (__int64)a2, v22);
                  if ( !(_BYTE)v5 )
                    goto LABEL_23;
                  v28 = (unsigned __int8)v5 & 2;
                  v29 = (unsigned __int8)v5 & 1;
                  if ( v29 )
                  {
                    if ( (_BYTE)v28 )
                    {
                      v30 = 3;
                      goto LABEL_29;
                    }
                  }
                  else
                  {
                    v30 = 2;
                    if ( (_BYTE)v28 )
                      goto LABEL_29;
                  }
                  v30 = v29;
LABEL_29:
                  LOBYTE(v5) = sub_FD9620((__int64)a1, v30, v25, v28, v26, v27, a3);
LABEL_23:
                  v24 += 32;
                  ++v22;
                  if ( v23 == v24 )
                    return (char)v5;
                }
              }
LABEL_57:
              BUG();
            }
          }
          v21 = 0;
          goto LABEL_21;
        }
      }
      else
      {
        v32 = *((_QWORD *)a2 - 4);
        if ( v32 && !*(_BYTE *)v32 && *(_QWORD *)(v32 + 24) == *((_QWORD *)a2 + 10) && *(_DWORD *)(v32 + 36) == 205 )
        {
          v34 = v9 & 1;
          v13 = 0;
          goto LABEL_15;
        }
      }
      v13 = 0;
      goto LABEL_15;
    }
LABEL_30:
    LOBYTE(v5) = sub_FD7FB0((__int64)a1, (__int64)a2);
    return (char)v5;
  }
  LOBYTE(v5) = sub_FD96C0((__int64)a1, (__int64)a2, a3);
  return (char)v5;
}
