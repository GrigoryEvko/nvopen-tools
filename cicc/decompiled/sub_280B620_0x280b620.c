// Function: sub_280B620
// Address: 0x280b620
//
void __fastcall sub_280B620(unsigned __int64 *a1)
{
  __int64 v1; // r13
  unsigned __int64 v3; // r15
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r15
  __int64 v14; // r13
  char v15; // dl
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  _QWORD *v28; // rbx
  _QWORD *v29; // r14
  __int64 v30; // rsi
  __int64 v31; // rax
  _QWORD *v32; // r14
  _QWORD *v33; // rbx
  __int64 v34; // rsi
  unsigned __int64 v35; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v36; // [rsp+10h] [rbp-A0h]
  _QWORD *v37; // [rsp+10h] [rbp-A0h]
  _QWORD *v38; // [rsp+10h] [rbp-A0h]
  _QWORD *v39; // [rsp+18h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  _QWORD *v41; // [rsp+18h] [rbp-98h]
  _QWORD *v42; // [rsp+18h] [rbp-98h]
  __int64 v43; // [rsp+28h] [rbp-88h] BYREF
  __int64 v44; // [rsp+30h] [rbp-80h]
  __int64 v45; // [rsp+38h] [rbp-78h]
  __int64 v46; // [rsp+40h] [rbp-70h]
  void *v47; // [rsp+50h] [rbp-60h]
  __int64 v48; // [rsp+58h] [rbp-58h] BYREF
  __int64 v49; // [rsp+60h] [rbp-50h]
  __int64 v50; // [rsp+68h] [rbp-48h]
  __int64 v51; // [rsp+70h] [rbp-40h]

  v1 = 0;
  v3 = *a1;
  if ( (unsigned __int64 *)*a1 != a1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = *(_BYTE *)(v3 + 128) ^ 1;
        if ( !v1 )
          break;
        if ( !v4 )
          goto LABEL_7;
        sub_280A780(v3 + 16, v1);
        v5 = *(_QWORD *)v3;
        --a1[2];
        v36 = v5;
        sub_2208CA0((__int64 *)v3);
        if ( *(_BYTE *)(v3 + 296) )
        {
          v27 = *(unsigned int *)(v3 + 288);
          *(_BYTE *)(v3 + 296) = 0;
          if ( (_DWORD)v27 )
          {
            v28 = *(_QWORD **)(v3 + 272);
            v29 = &v28[2 * v27];
            do
            {
              if ( *v28 != -8192 && *v28 != -4096 )
              {
                v30 = v28[1];
                if ( v30 )
                  sub_B91220((__int64)(v28 + 1), v30);
              }
              v28 += 2;
            }
            while ( v29 != v28 );
            LODWORD(v27) = *(_DWORD *)(v3 + 288);
          }
          sub_C7D6A0(*(_QWORD *)(v3 + 272), 16LL * (unsigned int)v27, 8);
        }
        v6 = *(unsigned int *)(v3 + 256);
        if ( (_DWORD)v6 )
        {
          v7 = *(_QWORD **)(v3 + 240);
          v43 = 2;
          v44 = 0;
          v45 = -4096;
          v39 = &v7[8 * v6];
          v8 = -4096;
          v46 = 0;
          v48 = 2;
          v49 = 0;
          v50 = -8192;
          v47 = &unk_49DD7B0;
          v51 = 0;
          while ( 1 )
          {
            v9 = v7[3];
            if ( v9 != v8 )
            {
              v8 = v50;
              if ( v9 != v50 )
              {
                v10 = v7[7];
                if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
                {
                  sub_BD60C0(v7 + 5);
                  v9 = v7[3];
                }
                v8 = v9;
              }
            }
            *v7 = &unk_49DB368;
            if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
              sub_BD60C0(v7 + 1);
            v7 += 8;
            if ( v39 == v7 )
              break;
            v8 = v45;
          }
          v47 = &unk_49DB368;
          if ( v50 != 0 && v50 != -4096 && v50 != -8192 )
            sub_BD60C0(&v48);
          if ( v45 != 0 && v45 != -4096 && v45 != -8192 )
            sub_BD60C0(&v43);
          LODWORD(v6) = *(_DWORD *)(v3 + 256);
        }
        sub_C7D6A0(*(_QWORD *)(v3 + 240), (unsigned __int64)(unsigned int)v6 << 6, 8);
        v11 = *(_QWORD *)(v3 + 152);
        if ( v11 != v3 + 168 )
          _libc_free(v11);
        v12 = *(_QWORD *)(v3 + 48);
        if ( v12 != v3 + 64 )
          _libc_free(v12);
        sub_C7D6A0(*(_QWORD *)(v3 + 24), 8LL * *(unsigned int *)(v3 + 40), 8);
        j_j___libc_free_0(v3);
        v3 = v36;
LABEL_4:
        if ( (unsigned __int64 *)v3 == a1 )
          goto LABEL_8;
      }
      if ( v4 )
      {
        v1 = v3 + 16;
        v3 = *(_QWORD *)v3;
        goto LABEL_4;
      }
LABEL_7:
      v3 = *(_QWORD *)v3;
      v1 = 0;
      if ( (unsigned __int64 *)v3 == a1 )
      {
LABEL_8:
        if ( (_BYTE)qword_4FFEE88 )
          return;
        v13 = *a1;
        if ( (unsigned __int64 *)*a1 == a1 )
          return;
        v14 = 0;
        while ( 2 )
        {
          v15 = *(_BYTE *)(v13 + 128);
          v40 = v13 + 16;
          if ( v15 )
          {
LABEL_48:
            if ( v14 )
            {
              if ( !v15 )
                goto LABEL_45;
              sub_280A780(v40, v14);
              v18 = *(_QWORD *)v13;
              --a1[2];
              v35 = v18;
              sub_2208CA0((__int64 *)v13);
              if ( *(_BYTE *)(v13 + 296) )
              {
                v31 = *(unsigned int *)(v13 + 288);
                *(_BYTE *)(v13 + 296) = 0;
                if ( (_DWORD)v31 )
                {
                  v32 = *(_QWORD **)(v13 + 272);
                  v33 = &v32[2 * v31];
                  do
                  {
                    if ( *v32 != -4096 && *v32 != -8192 )
                    {
                      v34 = v32[1];
                      if ( v34 )
                        sub_B91220((__int64)(v32 + 1), v34);
                    }
                    v32 += 2;
                  }
                  while ( v33 != v32 );
                  LODWORD(v31) = *(_DWORD *)(v13 + 288);
                }
                sub_C7D6A0(*(_QWORD *)(v13 + 272), 16LL * (unsigned int)v31, 8);
              }
              v19 = *(unsigned int *)(v13 + 256);
              if ( (_DWORD)v19 )
              {
                v20 = *(_QWORD **)(v13 + 240);
                v43 = 2;
                v44 = 0;
                v45 = -4096;
                v47 = &unk_49DD7B0;
                v21 = &v20[8 * v19];
                v22 = -4096;
                v46 = 0;
                v48 = 2;
                v49 = 0;
                v50 = -8192;
                v51 = 0;
                while ( 1 )
                {
                  v23 = v20[3];
                  if ( v23 != v22 )
                  {
                    v22 = v50;
                    if ( v23 != v50 )
                    {
                      v24 = v20[7];
                      if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
                      {
                        v37 = v21;
                        v41 = v20;
                        sub_BD60C0(v20 + 5);
                        v20 = v41;
                        v21 = v37;
                        v23 = v41[3];
                      }
                      v22 = v23;
                    }
                  }
                  *v20 = &unk_49DB368;
                  if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
                  {
                    v38 = v21;
                    v42 = v20;
                    sub_BD60C0(v20 + 1);
                    v21 = v38;
                    v20 = v42;
                  }
                  v20 += 8;
                  if ( v21 == v20 )
                    break;
                  v22 = v45;
                }
                v47 = &unk_49DB368;
                if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
                  sub_BD60C0(&v48);
                if ( v45 != -4096 && v45 != 0 && v45 != -8192 )
                  sub_BD60C0(&v43);
                LODWORD(v19) = *(_DWORD *)(v13 + 256);
              }
              sub_C7D6A0(*(_QWORD *)(v13 + 240), (unsigned __int64)(unsigned int)v19 << 6, 8);
              v25 = *(_QWORD *)(v13 + 152);
              if ( v25 != v13 + 168 )
                _libc_free(v25);
              v26 = *(_QWORD *)(v13 + 48);
              if ( v26 != v13 + 64 )
                _libc_free(v26);
              sub_C7D6A0(*(_QWORD *)(v13 + 24), 8LL * *(unsigned int *)(v13 + 40), 8);
              j_j___libc_free_0(v13);
              v13 = v35;
            }
            else
            {
              if ( !v15 )
                goto LABEL_45;
              v13 = *(_QWORD *)v13;
              v14 = v40;
            }
          }
          else
          {
            v16 = *(_QWORD *)(v13 + 48);
            v17 = v16 + 8LL * *(unsigned int *)(v13 + 56);
            if ( v16 != v17 )
            {
              while ( 1 )
              {
                if ( **(_BYTE **)v16 == 62 )
                {
                  v15 = sub_D364B0(*(_QWORD *)(*(_QWORD *)v16 + 40LL), a1[7], a1[9]);
                  if ( !v15 )
                    break;
                }
                v16 += 8;
                if ( v17 == v16 )
                  goto LABEL_48;
              }
            }
LABEL_45:
            v13 = *(_QWORD *)v13;
            v14 = 0;
          }
          if ( (unsigned __int64 *)v13 == a1 )
            return;
          continue;
        }
      }
    }
  }
}
