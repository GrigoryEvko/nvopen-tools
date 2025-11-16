// Function: sub_36F3880
// Address: 0x36f3880
//
__int64 __fastcall sub_36F3880(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 *v7; // r9
  __int64 v8; // rbx
  __int64 v9; // r13
  char v10; // al
  __int64 v11; // r15
  __int64 v12; // rax
  signed __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // r15
  __int64 *v17; // r12
  __int64 v18; // rax
  __int64 (__fastcall *v19)(__int64); // rax
  __int64 v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 i; // r15
  unsigned __int8 *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // r9
  __int64 *v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // [rsp+8h] [rbp-D8h]
  __int64 v43; // [rsp+10h] [rbp-D0h]
  __int64 *v44; // [rsp+10h] [rbp-D0h]
  int v45; // [rsp+18h] [rbp-C8h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  __int64 *v47; // [rsp+18h] [rbp-C8h]
  __int64 *v48; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+28h] [rbp-B8h]
  _BYTE v50[176]; // [rsp+30h] [rbp-B0h] BYREF

  v2 = a2;
  if ( (unsigned __int8)sub_CE9220(a1) )
  {
    if ( *(_DWORD *)(a2 + 1280) != 1 || (v28 = *(_QWORD *)(a1 + 80), v42 = a1 + 72, v28 == a1 + 72) )
    {
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
      {
LABEL_64:
        sub_B2C6D0(a1, a2, v4, v5);
        v8 = *(_QWORD *)(a1 + 96);
        v9 = v8 + 40LL * *(_QWORD *)(a1 + 104);
        if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        {
          sub_B2C6D0(a1, a2, v33, v34);
          v8 = *(_QWORD *)(a1 + 96);
        }
LABEL_6:
        while ( v9 != v8 )
        {
          v10 = *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL);
          if ( v10 == 14 )
          {
            if ( (unsigned __int8)sub_B2D680(v8) )
            {
              sub_36F20D0((_QWORD *)v2, v8);
            }
            else if ( *(_DWORD *)(v2 + 1280) == 1 )
            {
              v27 = *(_QWORD *)(v8 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 <= 1 )
                v27 = **(_QWORD **)(v27 + 16);
              if ( !(*(_DWORD *)(v27 + 8) >> 8) )
                sub_36F1860(v8, 1);
            }
          }
          else if ( v10 == 12 && *(_DWORD *)(v2 + 1280) == 1 )
          {
            v11 = *(_QWORD *)(v8 + 16);
            if ( v11 )
            {
              v12 = *(_QWORD *)(v8 + 16);
              while ( **(_BYTE **)(v12 + 24) == 77 )
              {
                v12 = *(_QWORD *)(v12 + 8);
                if ( !v12 )
                {
                  v13 = 0;
                  v48 = (__int64 *)v50;
                  v49 = 0x1000000000LL;
                  v14 = v11;
                  do
                  {
                    v14 = *(_QWORD *)(v14 + 8);
                    ++v13;
                  }
                  while ( v14 );
                  v15 = (__int64 *)v50;
                  if ( v13 > 16 )
                  {
                    v45 = v13;
                    sub_C8D5F0((__int64)&v48, v50, v13, 8u, v6, (__int64)v7);
                    LODWORD(v13) = v45;
                    v15 = &v48[(unsigned int)v49];
                  }
                  do
                  {
                    *v15++ = *(_QWORD *)(v11 + 24);
                    v11 = *(_QWORD *)(v11 + 8);
                  }
                  while ( v11 );
                  v16 = v48;
                  LODWORD(v49) = v49 + v13;
                  v17 = &v48[(unsigned int)v49];
                  if ( v48 != v17 )
                  {
                    do
                    {
                      v18 = *(_QWORD *)(*v16 + 8);
                      if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
                        v18 = **(_QWORD **)(v18 + 16);
                      if ( !(*(_DWORD *)(v18 + 8) >> 8) )
                        sub_36F1860(*v16, 1);
                      ++v16;
                    }
                    while ( v17 != v16 );
                    v17 = v48;
                  }
                  if ( v17 != (__int64 *)v50 )
                    _libc_free((unsigned __int64)v17);
                  break;
                }
              }
            }
          }
          v8 += 40;
        }
        return 1;
      }
    }
    else
    {
      do
      {
        if ( !v28 )
          BUG();
        for ( i = *(_QWORD *)(v28 + 32); v28 + 24 != i; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          if ( *(_BYTE *)(i - 24) == 61 && (*(_BYTE *)(*(_QWORD *)(i - 16) + 8LL) & 0xFD) == 0xC )
          {
            a2 = 6;
            v30 = sub_98ACB0(*(unsigned __int8 **)(i - 56), 6u);
            if ( *v30 == 22 )
            {
              if ( (unsigned __int8)sub_B2D680((__int64)v30) )
              {
                v31 = *(_QWORD *)(i - 16);
                if ( *(_BYTE *)(v31 + 8) == 14 )
                {
                  if ( !(*(_DWORD *)(v31 + 8) >> 8) )
                  {
                    a2 = 1;
                    sub_36F1860(i - 24, 1);
                  }
                }
                else
                {
                  v32 = *(_QWORD *)(i - 8);
                  if ( v32 )
                  {
                    v4 = *(_QWORD *)(i - 8);
                    while ( 1 )
                    {
                      v5 = *(_QWORD *)(v4 + 24);
                      if ( *(_BYTE *)v5 != 77 )
                        break;
                      v4 = *(_QWORD *)(v4 + 8);
                      if ( !v4 )
                      {
                        v37 = *(_QWORD *)(i - 8);
                        v38 = 0;
                        v48 = (__int64 *)v50;
                        a2 = 0x1000000000LL;
                        v49 = 0x1000000000LL;
                        do
                        {
                          v37 = *(_QWORD *)(v37 + 8);
                          ++v38;
                        }
                        while ( v37 );
                        v39 = (__int64 *)v50;
                        if ( v38 > 16 )
                        {
                          a2 = (__int64)v50;
                          v43 = v32;
                          v46 = v38;
                          sub_C8D5F0((__int64)&v48, v50, v38, 8u, v6, v38);
                          v32 = v43;
                          v38 = v46;
                          v39 = &v48[(unsigned int)v49];
                        }
                        do
                        {
                          *v39++ = *(_QWORD *)(v32 + 24);
                          v32 = *(_QWORD *)(v32 + 8);
                        }
                        while ( v32 );
                        v40 = v48;
                        v4 = v38 + (unsigned int)v49;
                        v5 = (unsigned int)v4;
                        LODWORD(v49) = v38 + v49;
                        v7 = &v48[(unsigned int)v4];
                        if ( v48 != v7 )
                        {
                          do
                          {
                            v41 = *(_QWORD *)(*v40 + 8);
                            v5 = (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17;
                            if ( (unsigned int)v5 <= 1 )
                              v41 = **(_QWORD **)(v41 + 16);
                            v4 = *(_DWORD *)(v41 + 8) >> 8;
                            if ( !(_DWORD)v4 )
                            {
                              a2 = 1;
                              v44 = v7;
                              v47 = v40;
                              sub_36F1860(*v40, 1);
                              v7 = v44;
                              v40 = v47;
                            }
                            ++v40;
                          }
                          while ( v7 != v40 );
                          v7 = v48;
                        }
                        if ( v7 != (__int64 *)v50 )
                          _libc_free((unsigned __int64)v7);
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        v28 = *(_QWORD *)(v28 + 8);
      }
      while ( v42 != v28 );
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        goto LABEL_64;
    }
    v8 = *(_QWORD *)(a1 + 96);
    v9 = v8 + 40LL * *(_QWORD *)(a1 + 104);
    goto LABEL_6;
  }
  v19 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(a2 + 1288) + 144LL);
  if ( v19 == sub_3020010 )
  {
    v20 = a2 + 2248;
    if ( (*(_BYTE *)(a1 + 2) & 1) == 0 )
    {
LABEL_30:
      v21 = *(_QWORD *)(a1 + 96);
      v22 = v21 + 40LL * *(_QWORD *)(a1 + 104);
      goto LABEL_31;
    }
  }
  else
  {
    v20 = v19(a2 + 1288);
    if ( (*(_BYTE *)(a1 + 2) & 1) == 0 )
      goto LABEL_30;
  }
  sub_B2C6D0(a1, a2, v4, v5);
  v21 = *(_QWORD *)(a1 + 96);
  v22 = v21 + 40LL * *(_QWORD *)(a1 + 104);
  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a1, a2, v35, v36);
    v21 = *(_QWORD *)(a1 + 96);
  }
LABEL_31:
  while ( v22 != v21 )
  {
    while ( *(_BYTE *)(*(_QWORD *)(v21 + 8) + 8LL) != 14 || !(unsigned __int8)sub_B2D680(v21) )
    {
      v21 += 40;
      if ( v22 == v21 )
        return 1;
    }
    v23 = *(_QWORD *)(v21 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 <= 1 )
      v23 = **(_QWORD **)(v23 + 16);
    if ( !(*(_DWORD *)(v23 + 8) >> 8) )
      sub_36F1860(v21, 5);
    v24 = v21;
    v25 = v21;
    v21 += 40;
    sub_36F1310(v25, v24, v20);
  }
  return 1;
}
