// Function: sub_1C0B430
// Address: 0x1c0b430
//
void __fastcall sub_1C0B430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6)
{
  __int64 v7; // r15
  __int64 v8; // rdi
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rbx
  bool v12; // r14
  __int64 v13; // r13
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // rax
  bool v17; // al
  bool v18; // r13
  __int64 v19; // r14
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // r12
  unsigned __int64 v23; // r13
  _QWORD *v24; // rax
  unsigned __int8 v25; // dl
  __int64 v26; // rdi
  char v27; // al
  int v28; // eax
  _QWORD *v29; // rdx
  _QWORD *v30; // rdx
  unsigned int v31; // r12d
  __int64 v32; // rsi
  __int64 *v33; // rax
  __int64 *v34; // rdi
  unsigned int v35; // r9d
  __int64 *v36; // rcx
  int v40; // [rsp+3Ch] [rbp-B4h]
  __int64 v41; // [rsp+40h] [rbp-B0h]
  __int64 v42; // [rsp+48h] [rbp-A8h]
  __int64 v43; // [rsp+58h] [rbp-98h] BYREF
  char v44[48]; // [rsp+60h] [rbp-90h] BYREF
  _BYTE *v45; // [rsp+90h] [rbp-60h] BYREF
  __int64 v46; // [rsp+98h] [rbp-58h]
  _BYTE v47[80]; // [rsp+A0h] [rbp-50h] BYREF

  *a6 = 0;
  v7 = *(_QWORD *)(a2 + 80);
  v45 = v47;
  v46 = 0x400000000LL;
  v42 = a2 + 72;
  v40 = 0;
  if ( v7 == a2 + 72 )
    return;
LABEL_4:
  while ( 2 )
  {
    v8 = v7 - 24;
    if ( !v7 )
      v8 = 0;
    v43 = v8;
    if ( (unsigned __int8)sub_1C089B0(v8, a4) )
    {
      v11 = *(_QWORD *)(v43 + 48);
      if ( v11 != v43 + 40 )
      {
        v41 = a4;
        v12 = 0;
        v13 = v43 + 40;
        while ( 1 )
        {
          while ( 1 )
          {
            if ( !v11 )
              BUG();
            v15 = *(_BYTE *)(v11 - 8);
            if ( v15 == 53 )
            {
              v14 = (unsigned int)v46;
              if ( (unsigned int)v46 >= HIDWORD(v46) )
              {
                sub_16CD150((__int64)&v45, v47, 0, 8, v9, v10);
                v14 = (unsigned int)v46;
              }
              *(_QWORD *)&v45[8 * v14] = v11 - 24;
              LODWORD(v46) = v46 + 1;
              goto LABEL_12;
            }
            if ( v15 != 55 )
              break;
            v16 = **(_QWORD **)(v11 - 48);
            if ( *(_BYTE *)(v16 + 8) == 16 )
              v16 = **(_QWORD **)(v16 + 16);
            v11 = *(_QWORD *)(v11 + 8);
            v17 = *(_DWORD *)(v16 + 8) >> 8 == 5 || *(_DWORD *)(v16 + 8) >> 8 == 0;
            if ( v17 )
              v12 = v17;
            if ( v13 == v11 )
            {
LABEL_21:
              v18 = v12;
              a4 = v41;
              if ( !v18 )
                goto LABEL_3;
              sub_1C0B2E0((__int64)v44, a5, &v43);
              v7 = *(_QWORD *)(v7 + 8);
              if ( v42 == v7 )
                goto LABEL_23;
              goto LABEL_4;
            }
          }
          if ( v15 == 78 )
          {
            v26 = *(_QWORD *)(v11 - 48);
            v27 = *(_BYTE *)(v26 + 16);
            if ( v27 != 20 )
            {
              if ( v27 || (*(_BYTE *)(v26 + 33) & 0x20) == 0 )
              {
                if ( (unsigned __int8)sub_1C07900(v11 - 24) )
                  goto LABEL_12;
                v28 = 7;
              }
              else
              {
                v31 = *(_DWORD *)(v26 + 36);
                if ( (unsigned __int8)sub_1C301F0(v31) )
                {
                  v32 = *(_QWORD *)(v11 + 16);
                  v33 = *(__int64 **)(a3 + 8);
                  if ( *(__int64 **)(a3 + 16) != v33 )
                    goto LABEL_66;
                  v34 = &v33[*(unsigned int *)(a3 + 28)];
                  v35 = *(_DWORD *)(a3 + 28);
                  if ( v33 != v34 )
                  {
                    v36 = 0;
                    while ( v32 != *v33 )
                    {
                      if ( *v33 == -2 )
                        v36 = v33;
                      if ( v34 == ++v33 )
                      {
                        if ( !v36 )
                          goto LABEL_75;
                        *v36 = v32;
                        --*(_DWORD *)(a3 + 32);
                        ++*(_QWORD *)a3;
                        goto LABEL_59;
                      }
                    }
                    goto LABEL_59;
                  }
LABEL_75:
                  if ( v35 < *(_DWORD *)(a3 + 24) )
                  {
                    *(_DWORD *)(a3 + 28) = v35 + 1;
                    *v34 = v32;
                    ++*(_QWORD *)a3;
                  }
                  else
                  {
LABEL_66:
                    sub_16CCBA0(a3, v32);
                  }
                }
LABEL_59:
                v28 = sub_1C07700(v31);
                if ( !v28 )
                  goto LABEL_12;
              }
LABEL_40:
              v40 |= v28;
              goto LABEL_12;
            }
            v28 = sub_1C090D0(v26);
            if ( v28 )
              goto LABEL_40;
          }
LABEL_12:
          v11 = *(_QWORD *)(v11 + 8);
          if ( v13 == v11 )
            goto LABEL_21;
        }
      }
    }
LABEL_3:
    v7 = *(_QWORD *)(v7 + 8);
    if ( v42 != v7 )
      continue;
    break;
  }
LABEL_23:
  if ( v40 && (_DWORD)v46 )
  {
    v19 = 8LL * (unsigned int)v46;
    v20 = 0;
    while ( 1 )
    {
      v21 = *(_QWORD *)&v45[v20];
      v22 = sub_22077B0(64);
      v23 = v22 + 24;
      *(_QWORD *)(v22 + 24) = sub_22077B0(512);
LABEL_27:
      v21 = *(_QWORD *)(v21 + 8);
      if ( v21 )
        break;
LABEL_45:
      j___libc_free_0(0);
      if ( v23 < v22 + 32 )
        j_j___libc_free_0(*(_QWORD *)(v22 + 24), 512);
      v20 += 8;
      j_j___libc_free_0(v22, 64);
      if ( v19 == v20 )
        goto LABEL_33;
    }
    while ( 1 )
    {
      v24 = sub_1648700(v21);
      v25 = *((_BYTE *)v24 + 16);
      if ( v25 <= 0x17u )
        goto LABEL_27;
      if ( v25 == 78 )
        goto LABEL_30;
      if ( v25 != 55 )
        break;
      v29 = (_QWORD *)*(v24 - 3);
      if ( !v29 || v24 != v29 )
        goto LABEL_30;
      v21 = *(_QWORD *)(v21 + 8);
      if ( !v21 )
        goto LABEL_45;
    }
    switch ( v25 )
    {
      case ':':
        v30 = (_QWORD *)*(v24 - 9);
        if ( v24 == v30 )
        {
LABEL_51:
          if ( v30 )
            goto LABEL_27;
        }
        break;
      case ';':
        v30 = (_QWORD *)*(v24 - 6);
        if ( v24 == v30 )
          goto LABEL_51;
        break;
      case 'E':
        break;
      default:
        goto LABEL_27;
    }
LABEL_30:
    j___libc_free_0(0);
    if ( v23 < v22 + 32 )
      j_j___libc_free_0(*(_QWORD *)(v22 + 24), 512);
    j_j___libc_free_0(v22, 64);
    *a6 = v40;
  }
LABEL_33:
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
}
