// Function: sub_E000A0
// Address: 0xe000a0
//
__int64 __fastcall sub_E000A0(unsigned __int8 *a1, __int64 a2)
{
  char v3; // r8
  __int64 result; // rax
  __int64 v5; // rsi
  bool v6; // al
  unsigned __int8 v7; // r9
  bool v8; // r8
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // r10d
  _BYTE *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int16 v20; // r10
  char v21; // r8

  v3 = sub_E00080(a1);
  result = 3;
  if ( v3 )
  {
    v5 = *(_QWORD *)(a2 + 16);
    if ( v5 )
    {
      v6 = sub_DFF600(*(_QWORD *)(a2 + 16));
      v7 = *(_BYTE *)(v5 - 16);
      v8 = (v7 & 2) != 0;
      if ( !v6 )
      {
        if ( (*(_BYTE *)(v5 - 16) & 2) != 0 )
        {
          if ( *(_DWORD *)(v5 - 24) <= 2u )
          {
            if ( !sub_DFF600(v5) )
              return 3;
LABEL_16:
            v13 = *(_DWORD *)(v5 - 24);
            if ( v13 <= 3 )
              return 3;
            v14 = *(_BYTE **)(*(_QWORD *)(v5 - 32) + 8LL);
            if ( !v14 || (unsigned __int8)(*v14 - 5) > 0x1Fu )
            {
              v15 = 4;
LABEL_20:
              if ( v13 < (unsigned int)((_DWORD)v15 != 3) + 4 )
                return 3;
              v16 = *(_QWORD *)(v5 - 32);
              goto LABEL_22;
            }
LABEL_34:
            if ( !sub_DFF670((__int64)v14) )
            {
              v15 = 3;
              if ( v21 )
                goto LABEL_20;
LABEL_38:
              v16 = v5 - 8LL * ((v7 >> 2) & 0xF) - 16;
LABEL_22:
              v17 = *(_QWORD *)(v16 + 8 * v15);
              if ( *(_BYTE *)v17 == 1 )
              {
                v18 = *(_QWORD *)(v17 + 136);
                if ( *(_BYTE *)v18 == 17 )
                {
                  v19 = *(_DWORD *)(v18 + 32) <= 0x40u ? *(_QWORD *)(v18 + 24) : **(_QWORD **)(v18 + 24);
                  if ( (v19 & 1) != 0 )
                    return 0;
                }
              }
              return 3;
            }
            v15 = 4;
            if ( v21 )
              goto LABEL_20;
LABEL_36:
            if ( v13 <= 4 )
              return 3;
            v15 = 4;
            goto LABEL_38;
          }
          v9 = *(_QWORD *)(v5 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v5 - 16) >> 6) & 0xFu) <= 2 )
          {
            if ( !sub_DFF600(v5) )
              return 3;
            goto LABEL_31;
          }
          v9 = v5 - 8LL * ((v7 >> 2) & 0xF) - 16;
        }
        v10 = *(_QWORD *)(v9 + 16);
        if ( *(_BYTE *)v10 == 1 )
        {
          v11 = *(_QWORD *)(v10 + 136);
          if ( *(_BYTE *)v11 == 17 )
          {
            v12 = *(_DWORD *)(v11 + 32) <= 0x40u ? *(_QWORD *)(v11 + 24) : **(_QWORD **)(v11 + 24);
            if ( (v12 & 1) != 0 )
              return 0;
          }
        }
        if ( !sub_DFF600(v5) )
          return 3;
      }
      if ( v8 )
        goto LABEL_16;
      v20 = *(_WORD *)(v5 - 16);
LABEL_31:
      v13 = (v20 >> 6) & 0xF;
      if ( v13 <= 3 )
        return 3;
      v14 = *(_BYTE **)(v5 - 8LL * ((v7 >> 2) & 0xF) - 8);
      if ( !v14 || (unsigned __int8)(*v14 - 5) > 0x1Fu )
        goto LABEL_36;
      goto LABEL_34;
    }
  }
  return result;
}
