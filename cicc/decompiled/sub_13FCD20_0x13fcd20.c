// Function: sub_13FCD20
// Address: 0x13fcd20
//
__int64 __fastcall sub_13FCD20(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 i; // r12
  char v13; // si
  unsigned int v14; // ebx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  int v21; // eax
  bool v22; // al
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rax
  unsigned int v28; // ebx
  __int64 v30; // rax
  int v31; // [rsp+4h] [rbp-3Ch]
  __int64 v32; // [rsp+8h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 32);
  v2 = *v1;
  v3 = *(_QWORD *)(*v1 + 8);
  if ( !v3 )
LABEL_53:
    BUG();
  while ( 1 )
  {
    v4 = sub_1648700(v3);
    v3 = *(_QWORD *)(v3 + 8);
    v5 = v4;
    if ( (unsigned __int8)(*(_BYTE *)(v4 + 16) - 25) <= 9u )
      break;
    if ( !v3 )
      goto LABEL_53;
  }
  if ( v3 )
  {
    while ( 1 )
    {
      v6 = sub_1648700(v3);
      if ( (unsigned __int8)(*(_BYTE *)(v6 + 16) - 25) <= 9u )
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
      if ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v3) + 16) - 25) <= 9u )
        return 0;
    }
    v9 = *(_QWORD *)(v6 + 40);
    v10 = a1 + 56;
    v11 = *(_QWORD *)(v5 + 40);
    if ( sub_1377F70(a1 + 56, v9) )
    {
      if ( !sub_1377F70(v10, v11) )
      {
LABEL_15:
        for ( i = *(_QWORD *)(v2 + 48); ; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          if ( *(_BYTE *)(i - 8) != 77 )
            return 0;
          v7 = i - 24;
          v13 = *(_BYTE *)(i - 1) & 0x40;
          v14 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
          if ( v14 )
          {
            v15 = 24LL * *(unsigned int *)(i + 32) + 8;
            v16 = 0;
            while ( 1 )
            {
              v17 = v7 - 24LL * v14;
              if ( v13 )
                v17 = *(_QWORD *)(i - 32);
              if ( v11 == *(_QWORD *)(v17 + v15) )
                break;
              ++v16;
              v15 += 8;
              if ( v14 == (_DWORD)v16 )
                goto LABEL_37;
            }
            v18 = 24 * v16;
            if ( !v13 )
            {
LABEL_38:
              v19 = v7 - 24LL * v14;
              goto LABEL_26;
            }
          }
          else
          {
LABEL_37:
            v18 = 0x17FFFFFFE8LL;
            if ( !v13 )
              goto LABEL_38;
          }
          v19 = *(_QWORD *)(i - 32);
LABEL_26:
          v20 = *(_QWORD *)(v19 + v18);
          if ( *(_BYTE *)(v20 + 16) != 13 )
            continue;
          if ( *(_DWORD *)(v20 + 32) <= 0x40u )
          {
            v22 = *(_QWORD *)(v20 + 24) == 0;
          }
          else
          {
            v31 = *(_DWORD *)(v20 + 32);
            v32 = v19;
            v21 = sub_16A57B0(v20 + 24);
            v19 = v32;
            v22 = v31 == v21;
          }
          if ( !v22 )
            continue;
          v23 = 0x17FFFFFFE8LL;
          if ( v14 )
          {
            v24 = 0;
            do
            {
              if ( v9 == *(_QWORD *)(v19 + 24LL * *(unsigned int *)(i + 32) + 8 * v24 + 8) )
              {
                v23 = 24 * v24;
                goto LABEL_35;
              }
              ++v24;
            }
            while ( v14 != (_DWORD)v24 );
            v25 = *(_QWORD *)(v19 + 0x17FFFFFFE8LL);
            if ( *(_BYTE *)(v25 + 16) != 35 )
              continue;
          }
          else
          {
LABEL_35:
            v25 = *(_QWORD *)(v19 + v23);
            if ( *(_BYTE *)(v25 + 16) != 35 )
              continue;
          }
          if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
            v26 = *(_QWORD **)(v25 - 8);
          else
            v26 = (_QWORD *)(v25 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF));
          if ( v7 == *v26 )
          {
            v27 = v26[3];
            if ( *(_BYTE *)(v27 + 16) == 13 )
            {
              v28 = *(_DWORD *)(v27 + 32);
              if ( v28 <= 0x40 ? *(_QWORD *)(v27 + 24) == 1 : v28 - 1 == (unsigned int)sub_16A57B0(v27 + 24) )
                return v7;
            }
          }
        }
      }
    }
    else if ( sub_1377F70(v10, v11) )
    {
      v30 = v11;
      v11 = v9;
      v9 = v30;
      goto LABEL_15;
    }
  }
  return 0;
}
