// Function: sub_1733F40
// Address: 0x1733f40
//
char __fastcall sub_1733F40(_DWORD **a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  char v5; // al
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // eax
  __int64 *v10; // r12
  unsigned __int8 v11; // al
  unsigned int v12; // r14d
  bool v13; // al
  __int64 v14; // rax
  int v15; // eax
  _BYTE *v16; // rbx
  unsigned __int8 v17; // al
  unsigned int v18; // r12d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // eax
  int v26; // eax
  _DWORD *v27; // rcx
  __int64 v28; // rax
  unsigned int v29; // r12d
  bool v30; // al
  __int64 v31; // rax
  unsigned int v32; // ebx
  unsigned int v33; // r15d
  int v34; // r14d
  __int64 v35; // rax
  unsigned int v36; // esi
  int v37; // r12d
  unsigned int v38; // r14d
  __int64 v39; // rax
  char v40; // dl
  unsigned int v41; // r15d
  unsigned int v42; // [rsp+Ch] [rbp-34h]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 52 )
  {
    v7 = *(_QWORD *)(a2 - 48);
    v8 = *(_QWORD *)(v7 + 8);
    if ( v8 )
    {
      if ( !*(_QWORD *)(v8 + 8) )
      {
        a4 = *(unsigned __int8 *)(v7 + 16);
        if ( (unsigned __int8)(a4 - 75) <= 1u )
        {
          v9 = *(unsigned __int16 *)(v7 + 18);
          BYTE1(v9) &= ~0x80u;
          **a1 = v9;
          v10 = *(__int64 **)(a2 - 24);
          v11 = *((_BYTE *)v10 + 16);
          if ( v11 == 13 )
          {
            v12 = *((_DWORD *)v10 + 8);
            if ( v12 <= 0x40 )
            {
              a4 = 64 - v12;
              v13 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) == v10[3];
            }
            else
            {
              v13 = v12 == (unsigned int)sub_16A58F0((__int64)(v10 + 3));
            }
            if ( v13 )
              return 1;
            goto LABEL_15;
          }
          if ( *(_BYTE *)(*v10 + 8) != 16 || v11 > 0x10u )
          {
LABEL_15:
            v14 = v10[1];
            if ( !v14 || *(_QWORD *)(v14 + 8) || (unsigned __int8)(*((_BYTE *)v10 + 16) - 75) > 1u )
              return 0;
            v15 = *((unsigned __int16 *)v10 + 9);
            BYTE1(v15) &= ~0x80u;
            **a1 = v15;
            v16 = *(_BYTE **)(v4 - 48);
            v17 = v16[16];
            if ( v17 == 13 )
            {
              v18 = *((_DWORD *)v16 + 8);
              if ( v18 <= 0x40 )
                return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18) == *((_QWORD *)v16 + 3);
              else
                return v18 == (unsigned int)sub_16A58F0((__int64)(v16 + 24));
            }
            if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 16 || v17 > 0x10u )
              return 0;
            v31 = sub_15A1020(v16, a2, *(_QWORD *)v16, a4);
            if ( v31 && *(_BYTE *)(v31 + 16) == 13 )
            {
              v32 = *(_DWORD *)(v31 + 32);
              if ( v32 <= 0x40 )
                return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32) == *(_QWORD *)(v31 + 24);
              else
                return v32 == (unsigned int)sub_16A58F0(v31 + 24);
            }
            v37 = *(_QWORD *)(*(_QWORD *)v16 + 32LL);
            if ( v37 )
            {
              v38 = 0;
              while ( 1 )
              {
                v39 = sub_15A0A60((__int64)v16, v38);
                if ( !v39 )
                  break;
                v40 = *(_BYTE *)(v39 + 16);
                if ( v40 != 9 )
                {
                  if ( v40 != 13 )
                    return 0;
                  v41 = *(_DWORD *)(v39 + 32);
                  if ( v41 <= 0x40 )
                  {
                    if ( *(_QWORD *)(v39 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v41) )
                      return 0;
                  }
                  else if ( v41 != (unsigned int)sub_16A58F0(v39 + 24) )
                  {
                    return 0;
                  }
                }
                if ( v37 == ++v38 )
                  return 1;
              }
              return 0;
            }
            return 1;
          }
          v28 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, *v10, a4);
          if ( v28 && *(_BYTE *)(v28 + 16) == 13 )
          {
            v29 = *(_DWORD *)(v28 + 32);
            if ( v29 <= 0x40 )
            {
              a4 = 64 - v29;
              v30 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v29) == *(_QWORD *)(v28 + 24);
            }
            else
            {
              v30 = v29 == (unsigned int)sub_16A58F0(v28 + 24);
            }
            if ( v30 )
              return 1;
          }
          else
          {
            v33 = 0;
            v34 = *(_QWORD *)(*v10 + 32);
            if ( !v34 )
              return 1;
            while ( 1 )
            {
              a2 = v33;
              v35 = sub_15A0A60((__int64)v10, v33);
              if ( !v35 )
                break;
              a4 = *(unsigned __int8 *)(v35 + 16);
              if ( (_BYTE)a4 != 9 )
              {
                if ( (_BYTE)a4 != 13 )
                  break;
                v36 = *(_DWORD *)(v35 + 32);
                if ( v36 <= 0x40 )
                {
                  a4 = 64 - v36;
                  a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36);
                  if ( *(_QWORD *)(v35 + 24) != a2 )
                    break;
                }
                else
                {
                  v42 = *(_DWORD *)(v35 + 32);
                  a2 = v42;
                  if ( v42 != (unsigned int)sub_16A58F0(v35 + 24) )
                    break;
                }
              }
              if ( v34 == ++v33 )
                return 1;
            }
          }
        }
      }
    }
    v10 = *(__int64 **)(v4 - 24);
    goto LABEL_15;
  }
  if ( v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v19 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v20 = *(_QWORD *)(a2 - 24 * v19);
  v21 = *(_QWORD *)(v20 + 8);
  if ( v21 && !*(_QWORD *)(v21 + 8) && (unsigned __int8)(*(_BYTE *)(v20 + 16) - 75) <= 1u )
  {
    v26 = *(unsigned __int16 *)(v20 + 18);
    v27 = *a1;
    BYTE1(v26) &= ~0x80u;
    *v27 = v26;
    if ( !sub_1727B40(
            *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
            a2,
            *(_DWORD *)(a2 + 20) & 0xFFFFFFF,
            (__int64)v27) )
    {
      v19 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      goto LABEL_24;
    }
    return 1;
  }
LABEL_24:
  v22 = *(_QWORD *)(a2 + 24 * (1 - v19));
  v23 = *(_QWORD *)(v22 + 8);
  if ( !v23 )
    return 0;
  if ( *(_QWORD *)(v23 + 8) )
    return 0;
  v24 = *(unsigned __int8 *)(v22 + 16);
  if ( (unsigned __int8)(v24 - 75) > 1u )
    return 0;
  v25 = *(unsigned __int16 *)(v22 + 18);
  BYTE1(v25) &= ~0x80u;
  **a1 = v25;
  return sub_1727B40(
           *(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
           a2,
           4LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
           v24);
}
