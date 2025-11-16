// Function: sub_14AB3F0
// Address: 0x14ab3f0
//
_BOOL8 __fastcall sub_14AB3F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  char v5; // dl
  _BOOL8 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 *v9; // r14
  int v10; // r12d
  int *v11; // rax
  __int64 v12; // rcx
  int v13; // eax
  unsigned __int8 v14; // al
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // r15
  unsigned __int8 v18; // al
  __int64 v19; // rax
  char v20; // al
  __int16 v21; // ax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r15
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  char v43; // al
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+8h] [rbp-48h]
  int v48; // [rsp+14h] [rbp-3Ch]
  int v49; // [rsp+14h] [rbp-3Ch]
  __int64 v50; // [rsp+18h] [rbp-38h]

  v4 = a1;
  if ( *(_BYTE *)(a1 + 16) != 14 )
  {
    v9 = a2;
    v10 = a3;
    do
    {
      v11 = (int *)sub_16D40F0(qword_4FBB370);
      if ( v11 )
        v13 = *v11;
      else
        v13 = qword_4FBB370[2];
      if ( v13 == v10 )
        return 0;
      v14 = *(_BYTE *)(v4 + 16);
      if ( v14 <= 0x17u )
      {
        if ( v14 != 5 )
          return 0;
        v20 = *(_BYTE *)(*(_QWORD *)v4 + 8LL);
        if ( v20 == 16 )
          v20 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v4 + 16LL) + 8LL);
        if ( (unsigned __int8)(v20 - 1) <= 5u || (v21 = *(_WORD *)(v4 + 18), v21 == 52) )
        {
          if ( (*(_BYTE *)(v4 + 17) & 0x10) != 0 )
            return 1;
          v21 = *(_WORD *)(v4 + 18);
        }
        if ( v21 != 12 )
          return 0;
        v16 = 1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
        v17 = *(_QWORD *)(v4 + 24 * v16);
        if ( *(_BYTE *)(v17 + 16) == 14 )
        {
LABEL_20:
          if ( *(_QWORD *)(v17 + 32) == sub_16982C0(qword_4FBB370, a2, v16, v12) )
          {
            v29 = *(_QWORD *)(v17 + 40);
            if ( (*(_BYTE *)(v29 + 26) & 7) != 3 )
              return 0;
            v19 = v29 + 8;
          }
          else
          {
            if ( (*(_BYTE *)(v17 + 50) & 7) != 3 )
              return 0;
            v19 = v17 + 32;
          }
          return (*(_BYTE *)(v19 + 18) & 8) == 0;
        }
        if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) != 16 )
          return 0;
        v22 = *(_QWORD *)(v4 + 24 * v16);
        v23 = sub_15A1020(v22);
        if ( !v23 || (v50 = v23, *(_BYTE *)(v23 + 16) != 14) )
        {
          v49 = *(_QWORD *)(*(_QWORD *)v17 + 32LL);
          if ( !v49 )
            return 1;
          v39 = 0;
          while ( 1 )
          {
            v40 = sub_15A0A60(v17, v39);
            v42 = v40;
            if ( !v40 )
              goto LABEL_40;
            v43 = *(_BYTE *)(v40 + 16);
            if ( v43 != 9 )
            {
              if ( v43 != 14 )
                goto LABEL_40;
              v47 = v42;
              if ( *(_QWORD *)(v42 + 32) == sub_16982C0(v17, (unsigned int)v39, v41, v42) )
              {
                v45 = *(_QWORD *)(v47 + 40);
                if ( (*(_BYTE *)(v45 + 26) & 7) != 3 )
                  goto LABEL_40;
                v44 = v45 + 8;
              }
              else
              {
                if ( (*(_BYTE *)(v47 + 50) & 7) != 3 )
                  goto LABEL_40;
                v44 = v47 + 32;
              }
              if ( (*(_BYTE *)(v44 + 18) & 8) != 0 )
                goto LABEL_40;
            }
            v39 = (unsigned int)(v39 + 1);
            if ( v49 == (_DWORD)v39 )
              return 1;
          }
        }
        goto LABEL_37;
      }
      v12 = *(_QWORD *)v4;
      v15 = *(unsigned __int8 *)(*(_QWORD *)v4 + 8LL);
      if ( (_BYTE)v15 == 16 )
        v15 = *(unsigned __int8 *)(**(_QWORD **)(v12 + 16) + 8LL);
      v16 = (unsigned int)(v15 - 1);
      if ( (unsigned __int8)v16 <= 5u || v14 == 76 )
      {
        if ( (*(_BYTE *)(v4 + 17) & 0x10) != 0 )
          return 1;
        if ( v14 == 36 )
        {
LABEL_19:
          v17 = *(_QWORD *)(v4 - 24);
          v18 = *(_BYTE *)(v17 + 16);
          if ( v18 == 14 )
            goto LABEL_20;
          if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) != 16 || v18 > 0x10u )
            return 0;
          v22 = *(_QWORD *)(v4 - 24);
          v30 = sub_15A1020(v22);
          if ( !v30 || (v50 = v30, *(_BYTE *)(v30 + 16) != 14) )
          {
            v48 = *(_QWORD *)(*(_QWORD *)v17 + 32LL);
            if ( !v48 )
              return 1;
            v31 = 0;
            while ( 1 )
            {
              v32 = sub_15A0A60(v17, v31);
              v34 = v32;
              if ( !v32 )
                goto LABEL_40;
              v35 = *(_BYTE *)(v32 + 16);
              if ( v35 != 9 )
              {
                if ( v35 != 14 )
                  goto LABEL_40;
                v46 = v34;
                if ( *(_QWORD *)(v34 + 32) == sub_16982C0(v17, (unsigned int)v31, v33, v34) )
                {
                  v38 = *(_QWORD *)(v46 + 40);
                  if ( (*(_BYTE *)(v38 + 26) & 7) != 3 )
                    goto LABEL_40;
                  v36 = v38 + 8;
                }
                else
                {
                  if ( (*(_BYTE *)(v46 + 50) & 7) != 3 )
                    goto LABEL_40;
                  v36 = v46 + 32;
                }
                if ( (*(_BYTE *)(v36 + 18) & 8) != 0 )
                  goto LABEL_40;
              }
              v31 = (unsigned int)(v31 + 1);
              if ( v48 == (_DWORD)v31 )
                return 1;
            }
          }
LABEL_37:
          if ( *(_QWORD *)(v50 + 32) == sub_16982C0(v22, a2, v24, v25) )
          {
            v37 = *(_QWORD *)(v50 + 40);
            if ( (*(_BYTE *)(v37 + 26) & 7) == 3 )
            {
              v26 = v37 + 8;
              goto LABEL_39;
            }
          }
          else
          {
            v26 = v50 + 32;
            if ( (*(_BYTE *)(v50 + 50) & 7) == 3 )
            {
LABEL_39:
              if ( (*(_BYTE *)(v26 + 18) & 8) == 0 )
                return 1;
            }
          }
LABEL_40:
          v14 = *(_BYTE *)(v4 + 16);
          if ( v14 <= 0x17u )
            return 0;
        }
      }
      else if ( v14 == 36 )
      {
        goto LABEL_19;
      }
      if ( (unsigned __int8)(v14 - 65) <= 1u )
        return 1;
      if ( v14 != 78 )
        return 0;
      a2 = v9;
      a1 = v4 | 4;
      v27 = sub_14AB140(v4 | 4, v9);
      if ( v27 == 96 )
        return 1;
      if ( v27 != 196 )
        return 0;
      ++v10;
      v28 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
      a3 = 4 * v28;
      v4 = *(_QWORD *)(v4 - 24 * v28);
    }
    while ( *(_BYTE *)(v4 + 16) != 14 );
  }
  if ( *(_QWORD *)(v4 + 32) != sub_16982C0(a1, a2, a3, a4) )
  {
    v5 = *(_BYTE *)(v4 + 50);
    result = 1;
    v7 = v4 + 32;
    if ( (v5 & 7) != 3 )
      return result;
    return ((*(_BYTE *)(v7 + 18) >> 3) ^ 1) & 1;
  }
  v8 = *(_QWORD *)(v4 + 40);
  result = 1;
  if ( (*(_BYTE *)(v8 + 26) & 7) == 3 )
  {
    v7 = v8 + 8;
    return ((*(_BYTE *)(v7 + 18) >> 3) ^ 1) & 1;
  }
  return result;
}
