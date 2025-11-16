// Function: sub_28EB620
// Address: 0x28eb620
//
__int64 __fastcall sub_28EB620(int a1, __int64 a2)
{
  int v2; // r12d
  unsigned int v4; // r14d
  __int64 v5; // rdx
  __int64 v6; // rsi
  _QWORD *v7; // rdi
  __int64 v8; // r8
  _BYTE *v9; // rdx
  __int64 v10; // r15
  unsigned int v11; // ebx
  __int64 v12; // rdi
  int v13; // eax
  bool v14; // al
  __int64 v15; // rbx
  unsigned __int8 v16; // al
  unsigned int v17; // r15d
  int v18; // eax
  int v19; // eax
  __int64 v21; // rbx
  _BYTE *v22; // rax
  __int64 v23; // r15
  _BYTE *v24; // rax
  unsigned int v25; // ebx
  int v26; // eax
  bool v27; // al
  __int64 v28; // r8
  size_t v29; // rbx
  _QWORD *v30; // rax
  bool v31; // bl
  __int64 v32; // rsi
  __int64 v33; // rax
  unsigned int v34; // ebx
  int v35; // eax
  char *v36; // rsi
  bool v37; // r15
  __int64 v38; // rsi
  __int64 v39; // rax
  unsigned int v40; // r15d
  int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // [rsp+0h] [rbp-50h]
  _BYTE *v44; // [rsp+0h] [rbp-50h]
  __int64 v45; // [rsp+8h] [rbp-48h]
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+8h] [rbp-48h]
  _BYTE *v48; // [rsp+8h] [rbp-48h]
  int v49; // [rsp+10h] [rbp-40h]
  int v50; // [rsp+10h] [rbp-40h]
  _BYTE *v52; // [rsp+18h] [rbp-38h]
  _BYTE *v53; // [rsp+18h] [rbp-38h]
  __int64 v54; // [rsp+18h] [rbp-38h]
  _BYTE *v55; // [rsp+18h] [rbp-38h]
  __int64 v56; // [rsp+18h] [rbp-38h]

  v2 = *(_DWORD *)(a2 + 8);
  if ( !v2 )
    return 0;
  v4 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v8 = 16LL * v4;
      v9 = *(_BYTE **)(*(_QWORD *)a2 + v8 + 8);
      if ( *v9 != 59 )
        goto LABEL_3;
      v10 = *((_QWORD *)v9 - 8);
      if ( *(_BYTE *)v10 == 17 )
      {
        v11 = *(_DWORD *)(v10 + 32);
        if ( !v11 )
          goto LABEL_28;
        if ( v11 > 0x40 )
        {
          v45 = 16LL * v4;
          v12 = v10 + 24;
          v52 = *(_BYTE **)(*(_QWORD *)a2 + v8 + 8);
          goto LABEL_12;
        }
        v14 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) == *(_QWORD *)(v10 + 24);
LABEL_13:
        if ( !v14 )
          goto LABEL_14;
        goto LABEL_28;
      }
      v21 = *(_QWORD *)(v10 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 || *(_BYTE *)v10 > 0x15u )
        goto LABEL_14;
      v45 = 16LL * v4;
      v52 = *(_BYTE **)(*(_QWORD *)a2 + v8 + 8);
      v22 = sub_AD7630(*((_QWORD *)v9 - 8), 0, (__int64)v9);
      v9 = v52;
      v8 = v45;
      if ( !v22 || *v22 != 17 )
      {
        if ( *(_BYTE *)(v21 + 8) != 17 || (v49 = *(_DWORD *)(v21 + 32)) == 0 )
        {
LABEL_14:
          v15 = *((_QWORD *)v9 - 4);
          v16 = *(_BYTE *)v15;
          if ( *(_BYTE *)v15 != 17 )
            goto LABEL_30;
          goto LABEL_15;
        }
        v31 = 0;
        v32 = 0;
        while ( 1 )
        {
LABEL_49:
          v43 = v8;
          v48 = v9;
          v33 = sub_AD69F0((unsigned __int8 *)v10, v32);
          v9 = v48;
          v8 = v43;
          if ( !v33 )
            goto LABEL_14;
          if ( *(_BYTE *)v33 == 13 )
            goto LABEL_56;
          if ( *(_BYTE *)v33 != 17 )
            goto LABEL_14;
          v34 = *(_DWORD *)(v33 + 32);
          if ( v34 )
            break;
          v31 = 1;
          v32 = (unsigned int)(v32 + 1);
          if ( v49 == (_DWORD)v32 )
          {
LABEL_57:
            if ( v31 )
              goto LABEL_28;
            goto LABEL_14;
          }
        }
        if ( v34 <= 0x40 )
        {
          v31 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v34) == *(_QWORD *)(v33 + 24);
        }
        else
        {
          v35 = sub_C445E0(v33 + 24);
          v9 = v48;
          v8 = v43;
          v31 = v34 == v35;
        }
        if ( !v31 )
          goto LABEL_14;
LABEL_56:
        v32 = (unsigned int)(v32 + 1);
        if ( v49 == (_DWORD)v32 )
          goto LABEL_57;
        goto LABEL_49;
      }
      v11 = *((_DWORD *)v22 + 8);
      if ( v11 )
      {
        if ( v11 <= 0x40 )
        {
          v14 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) == *((_QWORD *)v22 + 3);
        }
        else
        {
          v12 = (__int64)(v22 + 24);
LABEL_12:
          v13 = sub_C445E0(v12);
          v9 = v52;
          v8 = v45;
          v14 = v11 == v13;
        }
        goto LABEL_13;
      }
LABEL_28:
      v15 = *((_QWORD *)v9 - 4);
      if ( !v15 )
      {
        v16 = MEMORY[0];
        if ( MEMORY[0] != 17 )
        {
LABEL_30:
          v23 = *(_QWORD *)(v15 + 8);
          v55 = v9;
          if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 > 1 || v16 > 0x15u )
            goto LABEL_3;
          v47 = v8;
          v24 = sub_AD7630(v15, 0, (__int64)v9);
          v8 = v47;
          v9 = v55;
          if ( !v24 || *v24 != 17 )
          {
            if ( *(_BYTE *)(v23 + 8) == 17 )
            {
              v50 = *(_DWORD *)(v23 + 32);
              if ( v50 )
              {
                v37 = 0;
                v38 = 0;
                while ( 1 )
                {
                  v56 = v8;
                  v44 = v9;
                  v39 = sub_AD69F0((unsigned __int8 *)v15, v38);
                  v8 = v56;
                  if ( !v39 )
                    break;
                  v9 = v44;
                  if ( *(_BYTE *)v39 != 13 )
                  {
                    if ( *(_BYTE *)v39 != 17 )
                      goto LABEL_3;
                    v40 = *(_DWORD *)(v39 + 32);
                    if ( v40 )
                    {
                      if ( v40 <= 0x40 )
                      {
                        v37 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v40) == *(_QWORD *)(v39 + 24);
                      }
                      else
                      {
                        v41 = sub_C445E0(v39 + 24);
                        v9 = v44;
                        v8 = v56;
                        v37 = v40 == v41;
                      }
                      if ( !v37 )
                        goto LABEL_3;
                    }
                    else
                    {
                      v37 = 1;
                    }
                  }
                  v38 = (unsigned int)(v38 + 1);
                  if ( v50 == (_DWORD)v38 )
                  {
                    if ( v37 )
                      goto LABEL_18;
                    goto LABEL_3;
                  }
                }
              }
            }
            goto LABEL_3;
          }
          v25 = *((_DWORD *)v24 + 8);
          if ( v25 )
          {
            if ( v25 <= 0x40 )
            {
              v27 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25) == *((_QWORD *)v24 + 3);
            }
            else
            {
              v26 = sub_C445E0((__int64)(v24 + 24));
              v9 = v55;
              v8 = v47;
              v27 = v25 == v26;
            }
            if ( !v27 )
              goto LABEL_3;
          }
          goto LABEL_18;
        }
LABEL_15:
        v17 = *(_DWORD *)(v15 + 32);
        if ( v17 )
        {
          if ( v17 <= 0x40 )
          {
            if ( *(_QWORD *)(v15 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) )
              goto LABEL_3;
          }
          else
          {
            v46 = v8;
            v53 = v9;
            v18 = sub_C445E0(v15 + 24);
            v9 = v53;
            v8 = v46;
            if ( v17 != v18 )
              goto LABEL_3;
          }
        }
LABEL_18:
        v15 = *((_QWORD *)v9 - 8);
        if ( !v15 )
          goto LABEL_3;
      }
      v6 = v4;
      v54 = v8;
      v19 = sub_28E8F30((__int64 *)a2, v4, (_BYTE *)v15);
      v8 = v54;
      if ( v19 != v4 )
      {
        if ( a1 == 28 )
        {
          v42 = *(_QWORD *)(v15 + 8);
          return sub_AD6530(v42, v6);
        }
        if ( a1 == 29 )
          return sub_AD62B0(*(_QWORD *)(v15 + 8));
      }
LABEL_3:
      v5 = *(unsigned int *)(a2 + 8);
      if ( v4 + 1 != (_DWORD)v5 )
      {
        v6 = *(_QWORD *)a2;
        v7 = (_QWORD *)(*(_QWORD *)a2 + v8);
        if ( *(_QWORD *)(*(_QWORD *)a2 + 16LL * (v4 + 1) + 8) == v7[1] )
          break;
      }
      ++v4;
LABEL_6:
      if ( v2 == v4 )
        return 0;
    }
    if ( (unsigned int)(a1 - 28) <= 1 )
    {
      v36 = (char *)(16LL * (unsigned int)v5 + v6);
      if ( v36 != (char *)(v7 + 2) )
      {
        memmove(v7, v7 + 2, v36 - (char *)(v7 + 2));
        LODWORD(v5) = *(_DWORD *)(a2 + 8);
      }
      --v2;
      *(_DWORD *)(a2 + 8) = v5 - 1;
      goto LABEL_6;
    }
    if ( v2 == 2 )
      break;
    v28 = v8 + 32;
    v29 = 16 * v5 - v28;
    if ( v6 + v28 != v6 + 16 * v5 )
    {
      v30 = memmove(v7, (const void *)(v6 + v28), v29);
      v6 = *(_QWORD *)a2;
      v7 = v30;
    }
    v2 -= 2;
    *(_DWORD *)(a2 + 8) = (__int64)((__int64)v7 + v29 - v6) >> 4;
    if ( v2 == v4 )
      return 0;
  }
  v42 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 8LL);
  return sub_AD6530(v42, v6);
}
