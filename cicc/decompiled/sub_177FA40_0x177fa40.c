// Function: sub_177FA40
// Address: 0x177fa40
//
__int64 __fastcall sub_177FA40(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r12
  unsigned __int8 v8; // r13
  bool v9; // al
  __int64 v10; // rax
  __m128i *v11; // rax
  __int64 result; // rax
  __int64 v13; // rax
  int v14; // r13d
  unsigned int v15; // r15d
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rax
  int v19; // eax
  int v20; // eax
  unsigned int v21; // r14d
  __int64 v22; // r13
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  _BYTE *v32; // r13
  __int64 v33; // rdx
  bool v34; // al
  __int64 v35; // rdx
  __int64 v36; // rax
  char v37; // dl
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // r13
  __int64 v42; // rax
  unsigned __int8 v43; // al
  __int64 v44; // rax
  unsigned int v45; // r13d
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  bool v49; // al
  unsigned int v50; // r15d
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // r15d
  __int64 v54; // rax
  char v55; // cl
  __int64 v56; // rax
  unsigned int v57; // r15d
  __int64 v58; // rax
  char v59; // cl
  __int64 v60; // rax
  int v61; // [rsp+Ch] [rbp-54h]
  int v62; // [rsp+Ch] [rbp-54h]
  int v63; // [rsp+Ch] [rbp-54h]
  int v64; // [rsp+Ch] [rbp-54h]
  __m128i v65; // [rsp+10h] [rbp-50h] BYREF
  __int64 v66; // [rsp+20h] [rbp-40h]

  v7 = a2;
  v8 = *(_BYTE *)(a1 + 16);
  if ( v8 != 13 )
  {
    v19 = v8;
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 || v8 > 0x10u )
    {
LABEL_20:
      if ( v8 != 47 )
      {
        if ( v8 != 5 )
        {
          if ( v8 <= 0x17u )
            return 0;
          goto LABEL_58;
        }
        v20 = *(unsigned __int16 *)(a1 + 18);
        if ( (_WORD)v20 != 23 )
          goto LABEL_23;
        v31 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
        a4 = 4 * v31;
        v32 = *(_BYTE **)(a1 - 24 * v31);
        if ( v32[16] == 13 )
        {
          if ( *((_DWORD *)v32 + 8) <= 0x40u )
          {
            v33 = *((_QWORD *)v32 + 3);
            if ( v33 )
            {
              a4 = v33 - 1;
              if ( (v33 & (v33 - 1)) == 0 )
                goto LABEL_42;
            }
LABEL_23:
            if ( v20 != 37 )
              goto LABEL_24;
            if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              v35 = *(_QWORD *)(a1 - 8);
            else
              v35 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
            v36 = *(_QWORD *)v35;
            v37 = *(_BYTE *)(*(_QWORD *)v35 + 16LL);
            if ( v37 == 47 )
            {
              v41 = *(_QWORD *)(v36 - 48);
              v43 = *(_BYTE *)(v41 + 16);
              if ( v43 == 13 )
                goto LABEL_81;
              if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) == 16 && v43 <= 0x10u )
              {
                v44 = sub_15A1020((_BYTE *)v41, a2, *(_QWORD *)v41, a4);
                if ( v44 && *(_BYTE *)(v44 + 16) == 13 )
                  goto LABEL_92;
                v64 = *(_QWORD *)(*(_QWORD *)v41 + 32LL);
                if ( !v64 )
                  goto LABEL_42;
                v57 = 0;
                while ( 1 )
                {
                  v58 = sub_15A0A60(v41, v57);
                  if ( !v58 )
                    break;
                  v59 = *(_BYTE *)(v58 + 16);
                  if ( v59 != 9 )
                  {
                    if ( v59 != 13 )
                      break;
                    if ( *(_DWORD *)(v58 + 32) > 0x40u )
                    {
                      if ( (unsigned int)sub_16A5940(v58 + 24) != 1 )
                        break;
                    }
                    else
                    {
                      v60 = *(_QWORD *)(v58 + 24);
                      if ( !v60 || (v60 & (v60 - 1)) != 0 )
                        break;
                    }
                  }
                  if ( v64 == ++v57 )
                    goto LABEL_42;
                }
              }
            }
            else if ( v37 == 5 && *(_WORD *)(v36 + 18) == 23 )
            {
              v38 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
              v39 = 4 * v38;
              v40 = -3 * v38;
              v41 = *(_QWORD *)(v36 + 8 * v40);
              if ( *(_BYTE *)(v41 + 16) == 13 )
              {
LABEL_81:
                if ( *(_DWORD *)(v41 + 32) <= 0x40u )
                {
                  v42 = *(_QWORD *)(v41 + 24);
                  if ( !v42 )
                    goto LABEL_24;
                  goto LABEL_83;
                }
                v49 = (unsigned int)sub_16A5940(v41 + 24) == 1;
                goto LABEL_111;
              }
              if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) != 16 )
                goto LABEL_24;
              v44 = sub_15A1020((_BYTE *)v41, a2, v40, v39);
              if ( !v44 || *(_BYTE *)(v44 + 16) != 13 )
              {
                v53 = 0;
                v63 = *(_DWORD *)(*(_QWORD *)v41 + 32LL);
                while ( v63 != v53 )
                {
                  v54 = sub_15A0A60(v41, v53);
                  if ( !v54 )
                    goto LABEL_24;
                  v55 = *(_BYTE *)(v54 + 16);
                  if ( v55 != 9 )
                  {
                    if ( v55 != 13 )
                      goto LABEL_24;
                    if ( *(_DWORD *)(v54 + 32) > 0x40u )
                    {
                      if ( (unsigned int)sub_16A5940(v54 + 24) != 1 )
                        goto LABEL_24;
                    }
                    else
                    {
                      v56 = *(_QWORD *)(v54 + 24);
                      if ( !v56 || (v56 & (v56 - 1)) != 0 )
                        goto LABEL_24;
                    }
                  }
                  ++v53;
                }
                goto LABEL_42;
              }
LABEL_92:
              if ( *(_DWORD *)(v44 + 32) <= 0x40u )
              {
                v42 = *(_QWORD *)(v44 + 24);
                if ( !v42 )
                  goto LABEL_24;
LABEL_83:
                if ( (v42 & (v42 - 1)) != 0 )
                  goto LABEL_24;
LABEL_42:
                v65.m128i_i64[1] = a1;
                v65.m128i_i64[0] = (__int64)sub_1780AE0;
                v10 = *(unsigned int *)(v7 + 8);
                v66 = 0;
                if ( (unsigned int)v10 >= *(_DWORD *)(v7 + 12) )
                {
                  sub_16CD150(v7, (const void *)(v7 + 16), 0, 24, a5, a6);
                  v10 = *(unsigned int *)(v7 + 8);
                }
                goto LABEL_7;
              }
              v49 = (unsigned int)sub_16A5940(v44 + 24) == 1;
LABEL_111:
              if ( !v49 )
                goto LABEL_24;
              goto LABEL_42;
            }
LABEL_24:
            if ( a3 != 6 && *(_BYTE *)(a1 + 16) == 79 )
            {
              v21 = a3 + 1;
              v22 = sub_177FA40(*(_QWORD *)(a1 - 48), v7, v21);
              if ( v22 )
              {
                if ( sub_177FA40(*(_QWORD *)(a1 - 24), v7, v21) )
                {
                  v65.m128i_i64[1] = a1;
                  v10 = *(unsigned int *)(v7 + 8);
                  v65.m128i_i64[0] = 0;
                  v66 = v22 - 1;
                  if ( (unsigned int)v10 >= *(_DWORD *)(v7 + 12) )
                  {
                    sub_16CD150(v7, (const void *)(v7 + 16), 0, 24, v23, v24);
                    v10 = *(unsigned int *)(v7 + 8);
                  }
                  goto LABEL_7;
                }
              }
            }
            return 0;
          }
          v34 = (unsigned int)sub_16A5940((__int64)(v32 + 24)) == 1;
          goto LABEL_72;
        }
        if ( *(_BYTE *)(*(_QWORD *)v32 + 8LL) != 16 )
          goto LABEL_23;
        v29 = sub_15A1020(v32, a2, *(_QWORD *)v32, a4);
        if ( !v29 || *(_BYTE *)(v29 + 16) != 13 )
        {
          v62 = *(_QWORD *)(*(_QWORD *)v32 + 32LL);
          if ( !v62 )
            goto LABEL_42;
          v50 = 0;
          while ( 1 )
          {
            a2 = v50;
            v51 = sub_15A0A60((__int64)v32, v50);
            if ( !v51 )
              goto LABEL_73;
            a4 = *(unsigned __int8 *)(v51 + 16);
            if ( (_BYTE)a4 != 9 )
            {
              if ( (_BYTE)a4 != 13 )
                goto LABEL_73;
              if ( *(_DWORD *)(v51 + 32) > 0x40u )
              {
                if ( (unsigned int)sub_16A5940(v51 + 24) != 1 )
                  goto LABEL_73;
              }
              else
              {
                v52 = *(_QWORD *)(v51 + 24);
                if ( !v52 )
                  goto LABEL_73;
                a4 = v52 - 1;
                if ( (v52 & (v52 - 1)) != 0 )
                  goto LABEL_73;
              }
            }
            if ( v62 == ++v50 )
              goto LABEL_42;
          }
        }
        if ( *(_DWORD *)(v29 + 32) > 0x40u )
        {
          v34 = (unsigned int)sub_16A5940(v29 + 24) == 1;
LABEL_72:
          if ( v34 )
            goto LABEL_42;
LABEL_73:
          v19 = *(unsigned __int8 *)(a1 + 16);
LABEL_53:
          if ( (unsigned __int8)v19 <= 0x17u )
          {
            if ( (_BYTE)v19 != 5 )
              return 0;
            v20 = *(unsigned __int16 *)(a1 + 18);
            goto LABEL_23;
          }
LABEL_58:
          v20 = v19 - 24;
          goto LABEL_23;
        }
LABEL_105:
        v48 = *(_QWORD *)(v29 + 24);
        if ( v48 && (v48 & (v48 - 1)) == 0 )
          goto LABEL_42;
        goto LABEL_73;
      }
      v27 = *(_QWORD *)(a1 - 48);
      v28 = *(unsigned __int8 *)(v27 + 16);
      if ( (_BYTE)v28 == 13 )
      {
        if ( *(_DWORD *)(v27 + 32) <= 0x40u )
        {
          v30 = *(_QWORD *)(v27 + 24);
          if ( v30 )
          {
            a4 = v30 - 1;
            if ( (v30 & (v30 - 1)) == 0 )
              goto LABEL_42;
          }
          goto LABEL_58;
        }
        if ( (unsigned int)sub_16A5940(v27 + 24) == 1 )
          goto LABEL_42;
      }
      else
      {
        a4 = *(_QWORD *)v27;
        if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) != 16 || (unsigned __int8)v28 > 0x10u )
          goto LABEL_58;
        v29 = sub_15A1020(*(_BYTE **)(a1 - 48), a2, v28, a4);
        if ( !v29 || *(_BYTE *)(v29 + 16) != 13 )
        {
          v61 = *(_QWORD *)(*(_QWORD *)v27 + 32LL);
          if ( !v61 )
            goto LABEL_42;
          v45 = 0;
          while ( 1 )
          {
            a2 = v45;
            v46 = sub_15A0A60(v27, v45);
            if ( !v46 )
              goto LABEL_73;
            a4 = *(unsigned __int8 *)(v46 + 16);
            if ( (_BYTE)a4 != 9 )
            {
              if ( (_BYTE)a4 != 13 )
                goto LABEL_73;
              if ( *(_DWORD *)(v46 + 32) > 0x40u )
              {
                if ( (unsigned int)sub_16A5940(v46 + 24) != 1 )
                  goto LABEL_73;
              }
              else
              {
                v47 = *(_QWORD *)(v46 + 24);
                if ( !v47 )
                  goto LABEL_73;
                a4 = v47 - 1;
                if ( (v47 & (v47 - 1)) != 0 )
                  goto LABEL_73;
              }
            }
            if ( v61 == ++v45 )
              goto LABEL_42;
          }
        }
        if ( *(_DWORD *)(v29 + 32) <= 0x40u )
          goto LABEL_105;
        if ( (unsigned int)sub_16A5940(v29 + 24) == 1 )
          goto LABEL_42;
        v8 = *(_BYTE *)(a1 + 16);
      }
      v19 = v8;
      goto LABEL_53;
    }
    v25 = sub_15A1020((_BYTE *)a1, a2, *(_QWORD *)a1, a4);
    if ( !v25 || *(_BYTE *)(v25 + 16) != 13 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( !v14 )
      {
LABEL_5:
        v65.m128i_i64[1] = a1;
        v65.m128i_i64[0] = (__int64)sub_177F890;
        v10 = *(unsigned int *)(v7 + 8);
        v66 = 0;
        if ( (unsigned int)v10 >= *(_DWORD *)(v7 + 12) )
        {
          sub_16CD150(v7, (const void *)(v7 + 16), 0, 24, a5, a6);
          v10 = *(unsigned int *)(v7 + 8);
        }
LABEL_7:
        v11 = (__m128i *)(*(_QWORD *)v7 + 24 * v10);
        *v11 = _mm_loadu_si128(&v65);
        v11[1].m128i_i64[0] = v66;
        result = (unsigned int)(*(_DWORD *)(v7 + 8) + 1);
        *(_DWORD *)(v7 + 8) = result;
        return result;
      }
      v15 = 0;
      while ( 1 )
      {
        a2 = v15;
        v16 = sub_15A0A60(a1, v15);
        if ( !v16 )
          goto LABEL_19;
        v17 = *(_BYTE *)(v16 + 16);
        if ( v17 != 9 )
        {
          if ( v17 != 13 )
            goto LABEL_19;
          if ( *(_DWORD *)(v16 + 32) > 0x40u )
          {
            if ( (unsigned int)sub_16A5940(v16 + 24) != 1 )
              goto LABEL_19;
          }
          else
          {
            v18 = *(_QWORD *)(v16 + 24);
            if ( !v18 || (v18 & (v18 - 1)) != 0 )
              goto LABEL_19;
          }
        }
        if ( v14 == ++v15 )
          goto LABEL_5;
      }
    }
    if ( *(_DWORD *)(v25 + 32) <= 0x40u )
    {
      v26 = *(_QWORD *)(v25 + 24);
      if ( v26 && (v26 & (v26 - 1)) == 0 )
        goto LABEL_5;
LABEL_19:
      v8 = *(_BYTE *)(a1 + 16);
      v19 = v8;
      goto LABEL_20;
    }
    v9 = (unsigned int)sub_16A5940(v25 + 24) == 1;
LABEL_4:
    if ( v9 )
      goto LABEL_5;
    goto LABEL_19;
  }
  if ( *(_DWORD *)(a1 + 32) > 0x40u )
  {
    v9 = (unsigned int)sub_16A5940(a1 + 24) == 1;
    goto LABEL_4;
  }
  v13 = *(_QWORD *)(a1 + 24);
  if ( v13 && (v13 & (v13 - 1)) == 0 )
    goto LABEL_5;
  return 0;
}
