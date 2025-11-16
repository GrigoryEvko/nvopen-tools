// Function: sub_1CED620
// Address: 0x1ced620
//
__int64 __fastcall sub_1CED620(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // r14
  _BOOL4 v7; // eax
  __int64 v8; // rdi
  __int64 v9; // r9
  char v10; // di
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // r14d
  __int64 v21; // rax
  unsigned int v22; // r14d

  v2 = *(_QWORD *)(a2 + 48);
  if ( !v2 )
    BUG();
  if ( *(_BYTE *)(v2 - 8) == 77 && (*(_DWORD *)(v2 - 4) & 0xFFFFFFF) == 2 )
  {
    v4 = v2 - 72;
    if ( (*(_BYTE *)(v2 - 1) & 0x40) != 0 )
      v4 = *(_QWORD *)(v2 - 32);
    v5 = *(_QWORD *)(v4 + 24LL * *(unsigned int *)(v2 + 32) + 8);
    v6 = *(_QWORD *)(v4 + 24LL * *(unsigned int *)(v2 + 32) + 16);
    v7 = sub_1377F70(a1 + 56, v5);
    v8 = a1 + 56;
    if ( v7 )
    {
      if ( !sub_1377F70(v8, v6) )
        goto LABEL_9;
    }
    else if ( sub_1377F70(v8, v6) )
    {
      v5 = v6;
LABEL_9:
      while ( *(_BYTE *)(v2 - 8) == 77 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v2 - 24) + 8LL) == 11 )
        {
          v9 = v2 - 24;
          v10 = *(_BYTE *)(v2 - 1) & 0x40;
          v11 = *(_DWORD *)(v2 - 4) & 0xFFFFFFF;
          if ( v11 )
          {
            v12 = 24LL * *(unsigned int *)(v2 + 32) + 8;
            v13 = 0;
            while ( 1 )
            {
              v14 = v9 - 24LL * v11;
              if ( v10 )
                v14 = *(_QWORD *)(v2 - 32);
              if ( v5 == *(_QWORD *)(v14 + v12) )
                break;
              ++v13;
              v12 += 8;
              if ( v11 == (_DWORD)v13 )
                goto LABEL_36;
            }
            v15 = 24 * v13;
          }
          else
          {
LABEL_36:
            v15 = 0x17FFFFFFE8LL;
          }
          v16 = v10 ? *(_QWORD *)(v2 - 32) : v9 - 24LL * v11;
          v17 = *(_QWORD *)(v16 + v15);
          if ( *(_BYTE *)(v17 + 16) == 35 )
          {
            if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
              v18 = *(__int64 **)(v17 - 8);
            else
              v18 = (__int64 *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
            v19 = *v18;
            if ( v9 == *v18 )
            {
              v21 = v18[3];
              if ( *(_BYTE *)(v21 + 16) == 13 )
              {
                v22 = *(_DWORD *)(v21 + 32);
                if ( v22 <= 0x40 )
                {
                  if ( *(_QWORD *)(v21 + 24) == 1 )
                    return 1;
                }
                else if ( (unsigned int)sub_16A57B0(v21 + 24) == v22 - 1 )
                {
                  return 1;
                }
              }
            }
            else if ( v9 == v18[3] && *(_BYTE *)(v19 + 16) == 13 )
            {
              v20 = *(_DWORD *)(v19 + 32);
              if ( v20 <= 0x40 )
              {
                if ( *(_QWORD *)(v19 + 24) == 1 )
                  return 1;
              }
              else if ( (unsigned int)sub_16A57B0(v19 + 24) == v20 - 1 )
              {
                return 1;
              }
            }
          }
        }
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          BUG();
      }
    }
  }
  return 0;
}
