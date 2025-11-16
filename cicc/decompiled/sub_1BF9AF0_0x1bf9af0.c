// Function: sub_1BF9AF0
// Address: 0x1bf9af0
//
__int64 __fastcall sub_1BF9AF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v5; // rax
  _QWORD *v6; // r13
  _QWORD *v7; // r14
  __int64 v8; // r14
  __int64 v10; // rbx
  __int64 v11; // r13
  _BOOL4 v12; // eax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rbx
  __int64 v16; // rax
  char v17; // di
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  bool v23; // al
  __int64 v24; // rax
  char v25; // di
  unsigned int v26; // esi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // [rsp+8h] [rbp-48h]
  int v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+18h] [rbp-38h]
  __int64 v40; // [rsp+18h] [rbp-38h]
  __int64 v41; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 8LL);
  v39 = **(_QWORD **)(a1 + 32);
  if ( !v3 )
LABEL_61:
    BUG();
  while ( 1 )
  {
    v5 = sub_1648700(v3);
    v3 = *(_QWORD *)(v3 + 8);
    v6 = v5;
    if ( (unsigned __int8)(*((_BYTE *)v5 + 16) - 25) <= 9u )
      break;
    if ( !v3 )
      goto LABEL_61;
  }
  if ( v3 )
  {
    while ( 1 )
    {
      v7 = sub_1648700(v3);
      if ( (unsigned __int8)(*((_BYTE *)v7 + 16) - 25) <= 9u )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return 0;
    }
    while ( 1 )
    {
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        break;
      if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v3) + 16) - 25) <= 9u )
        return 0;
    }
    v10 = a1 + 56;
    v11 = v6[5];
    v36 = v7[5];
    if ( sub_1377F70(a1 + 56, v36) )
    {
      v12 = sub_1377F70(v10, v11);
      v14 = v36;
      if ( !v12 )
        goto LABEL_15;
    }
    else if ( sub_1377F70(v10, v11) )
    {
      v34 = v11;
      v11 = v36;
      v14 = v34;
LABEL_15:
      v15 = *(_QWORD *)(v39 + 48);
      if ( a2 )
        v15 = *(_QWORD *)(a2 + 32);
      while ( 1 )
      {
        if ( !v15 )
          BUG();
        if ( *(_BYTE *)(v15 - 8) != 77 )
          return 0;
        if ( !a3 || a3 == *(_QWORD *)(v15 - 24) )
        {
          v8 = v15 - 24;
          v16 = 0x17FFFFFFE8LL;
          v17 = *(_BYTE *)(v15 - 1) & 0x40;
          v18 = *(_DWORD *)(v15 - 4) & 0xFFFFFFF;
          if ( (*(_DWORD *)(v15 - 4) & 0xFFFFFFF) != 0 )
          {
            v19 = 24LL * *(unsigned int *)(v15 + 32) + 8;
            v20 = 0;
            do
            {
              v13 = v8 - 24LL * (unsigned int)v18;
              if ( v17 )
                v13 = *(_QWORD *)(v15 - 32);
              if ( v11 == *(_QWORD *)(v13 + v19) )
              {
                v16 = 24 * v20;
                goto LABEL_30;
              }
              ++v20;
              v19 += 8;
            }
            while ( (_DWORD)v18 != (_DWORD)v20 );
            v16 = 0x17FFFFFFE8LL;
          }
LABEL_30:
          if ( v17 )
          {
            v21 = *(_QWORD *)(v15 - 32);
          }
          else
          {
            v18 = (unsigned int)v18;
            v13 = 24LL * (unsigned int)v18;
            v21 = v8 - v13;
          }
          v22 = *(_QWORD *)(v21 + v16);
          if ( *(_BYTE *)(v22 + 16) == 13 )
          {
            v40 = v14;
            v23 = sub_1593BB0(v22, v18, v21, v13);
            v14 = v40;
            if ( v23 )
            {
              v24 = 0x17FFFFFFE8LL;
              v25 = *(_BYTE *)(v15 - 1) & 0x40;
              v26 = *(_DWORD *)(v15 - 4) & 0xFFFFFFF;
              if ( v26 )
              {
                v27 = 24LL * *(unsigned int *)(v15 + 32) + 8;
                v28 = 0;
                do
                {
                  v13 = v8 - 24LL * v26;
                  if ( v25 )
                    v13 = *(_QWORD *)(v15 - 32);
                  if ( v40 == *(_QWORD *)(v13 + v27) )
                  {
                    v24 = 24 * v28;
                    goto LABEL_41;
                  }
                  ++v28;
                  v27 += 8;
                }
                while ( v26 != (_DWORD)v28 );
                v24 = 0x17FFFFFFE8LL;
              }
LABEL_41:
              if ( v25 )
              {
                v29 = *(_QWORD *)(v15 - 32);
              }
              else
              {
                v13 = 24LL * v26;
                v29 = v8 - v13;
              }
              v30 = *(_QWORD *)(v29 + v24);
              if ( *(_BYTE *)(v30 + 16) == 35 )
              {
                v31 = (*(_BYTE *)(v30 + 23) & 0x40) != 0
                    ? *(_QWORD **)(v30 - 8)
                    : (_QWORD *)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF));
                if ( v8 == *v31 )
                {
                  v32 = v31[3];
                  if ( *(_BYTE *)(v32 + 16) == 13 )
                  {
                    v13 = *(unsigned int *)(v32 + 32);
                    v37 = *(_DWORD *)(v32 + 32);
                    if ( (unsigned int)v13 > 0x40 )
                    {
                      v38 = v40;
                      v41 = v31[3];
                      v35 = sub_16A57B0(v32 + 24);
                      v14 = v38;
                      v13 = (unsigned int)(v37 - v35);
                      if ( (unsigned int)v13 > 0x40 )
                        goto LABEL_18;
                      v33 = **(_QWORD **)(v41 + 24);
                    }
                    else
                    {
                      v33 = *(_QWORD *)(v32 + 24);
                    }
                    if ( v33 == 1 )
                      return v8;
                  }
                }
              }
            }
          }
        }
LABEL_18:
        v15 = *(_QWORD *)(v15 + 8);
      }
    }
  }
  return 0;
}
