// Function: sub_17386E0
// Address: 0x17386e0
//
char __fastcall sub_17386E0(_QWORD **a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  char v5; // al
  bool v6; // r12
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  char v12; // al
  unsigned __int64 v13; // rsi
  bool v14; // zf
  int v15; // eax
  __int64 v16; // rcx
  _QWORD *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rcx
  char v21; // al
  unsigned __int64 v22; // rsi
  int v23; // eax
  unsigned __int8 v24; // al
  unsigned int v25; // r12d
  bool v26; // al
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rcx
  bool v30; // al
  __int64 v31; // rax
  unsigned int v32; // r12d
  bool v33; // al
  bool v34; // al
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned int v37; // r14d
  __int64 v38; // rax
  unsigned int v39; // esi
  __int64 v40; // rax
  int v41; // [rsp+8h] [rbp-38h]
  int v42; // [rsp+Ch] [rbp-34h]

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 52 )
  {
    v8 = *(_QWORD *)(a2 - 48);
    v9 = *(_QWORD *)(v8 + 8);
    if ( !v9 || *(_QWORD *)(v9 + 8) )
      goto LABEL_8;
    v21 = *(_BYTE *)(v8 + 16);
    if ( v21 == 51 )
    {
      v34 = sub_171DA10(a1, *(_QWORD *)(v8 - 48), (__int64)a3, a4);
      v22 = *(_QWORD *)(v8 - 24);
      if ( v34 && v22 )
      {
        *a1[2] = v22;
        goto LABEL_27;
      }
      if ( !sub_171DA10(a1, v22, (__int64)a3, a4) )
        goto LABEL_8;
      v35 = *(_QWORD *)(v8 - 48);
      if ( !v35 )
        goto LABEL_8;
    }
    else
    {
      if ( v21 != 5 || *(_WORD *)(v8 + 18) != 27 )
        goto LABEL_8;
      v22 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
      v14 = sub_14B2B20(a1, v22) == 0;
      v23 = *(_DWORD *)(v8 + 20);
      if ( !v14 )
      {
        a4 = v23 & 0xFFFFFFF;
        a3 = *(_QWORD **)(v8 + 24 * (1 - a4));
        if ( a3 )
        {
          *a1[2] = a3;
LABEL_27:
          v10 = *(_QWORD *)(a2 - 24);
          v24 = *(_BYTE *)(v10 + 16);
          if ( v24 == 13 )
          {
            v25 = *(_DWORD *)(v10 + 32);
            if ( v25 <= 0x40 )
            {
              a4 = 64 - v25;
              v26 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25) == *(_QWORD *)(v10 + 24);
            }
            else
            {
              v26 = v25 == (unsigned int)sub_16A58F0(v10 + 24);
            }
            if ( v26 )
              return 1;
LABEL_9:
            v11 = *(_QWORD *)(v10 + 8);
            if ( !v11 || *(_QWORD *)(v11 + 8) )
              return 0;
            v12 = *(_BYTE *)(v10 + 16);
            if ( v12 == 51 )
            {
              v30 = sub_171DA10(a1, *(_QWORD *)(v10 - 48), (__int64)a3, a4);
              v13 = *(_QWORD *)(v10 - 24);
              if ( v30 && v13 )
              {
                *a1[2] = v13;
                return sub_17279D0(*(_BYTE **)(a2 - 48), v13, (__int64)v17, v16);
              }
              if ( !sub_171DA10(a1, v13, (__int64)v17, v16) )
                return 0;
              v36 = *(_QWORD *)(v10 - 48);
              if ( !v36 )
                return 0;
            }
            else
            {
              if ( v12 != 5 || *(_WORD *)(v10 + 18) != 27 )
                return 0;
              v13 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
              v14 = sub_14B2B20(a1, v13) == 0;
              v15 = *(_DWORD *)(v10 + 20);
              if ( !v14 )
              {
                v16 = v15 & 0xFFFFFFF;
                v17 = *(_QWORD **)(v10 + 24 * (1 - v16));
                if ( v17 )
                {
                  *a1[2] = v17;
                  return sub_17279D0(*(_BYTE **)(a2 - 48), v13, (__int64)v17, v16);
                }
              }
              v13 = *(_QWORD *)(v10 + 24 * (1LL - (v15 & 0xFFFFFFF)));
              if ( !sub_14B2B20(a1, v13) )
                return 0;
              v36 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
              if ( !v36 )
                return 0;
            }
            v17 = a1[2];
            *v17 = v36;
            return sub_17279D0(*(_BYTE **)(a2 - 48), v13, (__int64)v17, v16);
          }
          a3 = *(_QWORD **)v10;
          v6 = v24 <= 0x10u && *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16;
          if ( !v6 )
            goto LABEL_9;
          v31 = sub_15A1020(*(_BYTE **)(a2 - 24), v22, (__int64)a3, a4);
          if ( v31 && *(_BYTE *)(v31 + 16) == 13 )
          {
            v32 = *(_DWORD *)(v31 + 32);
            if ( v32 <= 0x40 )
            {
              a4 = 64 - v32;
              a3 = (_QWORD *)(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32));
              v33 = a3 == *(_QWORD **)(v31 + 24);
            }
            else
            {
              v33 = v32 == (unsigned int)sub_16A58F0(v31 + 24);
            }
            if ( v33 )
              return 1;
          }
          else
          {
            v37 = 0;
            v42 = *(_QWORD *)(*(_QWORD *)v10 + 32LL);
            if ( !v42 )
              return v6;
            while ( 1 )
            {
              v38 = sub_15A0A60(v10, v37);
              if ( !v38 )
                break;
              a4 = *(unsigned __int8 *)(v38 + 16);
              if ( (_BYTE)a4 != 9 )
              {
                if ( (_BYTE)a4 != 13 )
                  break;
                v39 = *(_DWORD *)(v38 + 32);
                if ( v39 <= 0x40 )
                {
                  a4 = 64 - v39;
                  if ( *(_QWORD *)(v38 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v39) )
                    break;
                }
                else
                {
                  v41 = *(_DWORD *)(v38 + 32);
                  if ( v41 != (unsigned int)sub_16A58F0(v38 + 24) )
                    break;
                }
              }
              if ( v42 == ++v37 )
                return v6;
            }
          }
LABEL_8:
          v10 = *(_QWORD *)(a2 - 24);
          goto LABEL_9;
        }
      }
      v22 = *(_QWORD *)(v8 + 24 * (1LL - (v23 & 0xFFFFFFF)));
      if ( !sub_14B2B20(a1, v22) )
        goto LABEL_8;
      v40 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
      a3 = (_QWORD *)(4 * v40);
      v35 = *(_QWORD *)(v8 - 24 * v40);
      if ( !v35 )
        goto LABEL_8;
    }
    a3 = a1[2];
    *a3 = v35;
    goto LABEL_27;
  }
  if ( v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v18 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v19 = *(_QWORD *)(a2 - 24 * v18);
  if ( sub_17385C0(a1, v19, 4 * v18, a4)
    && sub_1727B40(
         *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
         v19,
         *(_DWORD *)(a2 + 20) & 0xFFFFFFF,
         v20) )
  {
    return 1;
  }
  v27 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v28 = *(_QWORD *)(a2 + 24 * (1 - v27));
  if ( !sub_17385C0(a1, v28, v27, v20) )
    return 0;
  return sub_1727B40(
           *(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
           v28,
           4LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
           v29);
}
