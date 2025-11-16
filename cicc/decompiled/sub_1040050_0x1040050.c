// Function: sub_1040050
// Address: 0x1040050
//
_QWORD *__fastcall sub_1040050(__int64 a1, __int64 a2, _QWORD *a3, _BYTE *a4)
{
  __int64 v7; // rbx
  _QWORD *v8; // r15
  _QWORD *v9; // r12
  __int64 v10; // rax
  int v11; // eax
  bool v12; // zf
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  _QWORD *v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  int v20; // eax
  unsigned int v21; // esi
  __int64 v22; // r14
  __int64 v23; // rdi
  int v24; // r11d
  __int64 *v25; // r9
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rax
  char v31; // al
  unsigned __int16 v32; // dx
  __int64 v33; // rax
  int v34; // r14d
  _QWORD *v35; // r15
  __int64 v36; // rax
  int v37; // eax
  _QWORD *v38; // rax
  __int64 v39; // rdx
  int v40; // edx
  __int64 v41; // rax
  unsigned int v42; // eax
  int v43; // esi
  __int64 v44; // rdx
  __int64 v45; // rdx
  int v46; // eax
  __int64 v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+18h] [rbp-78h] BYREF
  __m128i v50[3]; // [rsp+20h] [rbp-70h] BYREF
  char v51; // [rsp+50h] [rbp-40h]

  v7 = a2;
  if ( *(_BYTE *)a2 == 85 )
  {
    v41 = *(_QWORD *)(a2 - 32);
    if ( v41 )
    {
      if ( !*(_BYTE *)v41 && *(_QWORD *)(v41 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v41 + 33) & 0x20) != 0 )
      {
        v42 = *(_DWORD *)(v41 + 36);
        if ( v42 == 155 )
          return 0;
        if ( v42 > 0x9B )
        {
          if ( v42 == 291 )
            return 0;
        }
        else if ( v42 > 6 )
        {
          if ( v42 == 11 )
            return 0;
        }
        else if ( v42 > 4 )
        {
          return 0;
        }
      }
    }
  }
  if ( !(unsigned __int8)sub_B46420(a2) && !(unsigned __int8)sub_B46490(a2) )
    return 0;
  if ( a4 )
  {
    if ( (unsigned __int8)(*a4 - 26) <= 1u )
    {
      if ( *a4 != 27 )
      {
LABEL_6:
        v8 = (_QWORD *)sub_BD5C60(a2);
        v47 = *(_QWORD *)(a2 + 40);
        v9 = sub_BD2C40(88, unk_3F8E4D0);
        if ( v9 )
        {
          v10 = sub_BCB120(v8);
          sub_BD35F0((__int64)v9, v10, 26);
          v11 = *((_DWORD *)v9 + 1);
          v9[4] = 0;
          v9[5] = 0;
          v9[9] = a2;
          v9[6] = 0;
          v12 = *(_BYTE *)v9 == 26;
          *((_DWORD *)v9 + 1) = v11 & 0x38000000 | 1;
          v9[7] = 0;
          v9[3] = sub_103AB70;
          v9[8] = v47;
          v13 = v9 - 8;
          if ( v12 )
            v13 = v9 - 4;
          if ( *v13 )
          {
            v14 = v13[1];
            *(_QWORD *)v13[2] = v14;
            if ( v14 )
              *(_QWORD *)(v14 + 16) = v13[2];
          }
          *v13 = 0;
          *((_DWORD *)v9 + 20) = -1;
        }
        if ( *(_BYTE *)a2 == 61 )
        {
          if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && sub_B91C10(a2, 6)
            || (sub_D665A0(v50, a2), (sub_CF5020((__int64)a3, (__int64)v50, 0) & 2) == 0) )
          {
            v15 = *(v9 - 4);
            v16 = v9 - 4;
            v17 = *(_QWORD *)(a1 + 128);
            if ( *(_BYTE *)v9 == 27 )
            {
              if ( v15 )
              {
                v18 = *(v9 - 3);
                *(_QWORD *)*(v9 - 2) = v18;
                if ( v18 )
                  *(_QWORD *)(v18 + 16) = *(v9 - 2);
              }
              *(v9 - 4) = v17;
              if ( v17 )
              {
                v19 = *(_QWORD *)(v17 + 16);
                *(v9 - 3) = v19;
                if ( v19 )
                  *(_QWORD *)(v19 + 16) = v9 - 3;
                *(v9 - 2) = v17 + 16;
                *(_QWORD *)(v17 + 16) = v16;
              }
              if ( *(_BYTE *)v17 == 27 )
                v20 = *(_DWORD *)(v17 + 80);
              else
                v20 = *(_DWORD *)(v17 + 72);
              *((_DWORD *)v9 + 21) = v20;
            }
            else
            {
              if ( *(_BYTE *)v17 == 27 )
                v43 = *(_DWORD *)(v17 + 80);
              else
                v43 = *(_DWORD *)(v17 + 72);
              *((_DWORD *)v9 + 20) = v43;
              if ( v15 )
              {
                v44 = *(v9 - 3);
                *(_QWORD *)*(v9 - 2) = v44;
                if ( v44 )
                  *(_QWORD *)(v44 + 16) = *(v9 - 2);
              }
              *(v9 - 4) = v17;
              v45 = *(_QWORD *)(v17 + 16);
              *(v9 - 3) = v45;
              if ( v45 )
                *(_QWORD *)(v45 + 16) = v9 - 3;
              *(v9 - 2) = v17 + 16;
              *(_QWORD *)(v17 + 16) = v16;
            }
          }
        }
LABEL_27:
        v21 = *(_DWORD *)(a1 + 56);
        v49 = v7;
        v22 = a1 + 32;
        if ( v21 )
          goto LABEL_28;
LABEL_44:
        ++*(_QWORD *)(a1 + 32);
        v50[0].m128i_i64[0] = 0;
        goto LABEL_45;
      }
      goto LABEL_37;
    }
    return 0;
  }
  v51 = 0;
  v31 = sub_CF6520(a3, (unsigned __int8 *)a2, v50);
  if ( (v31 & 2) == 0 )
  {
    if ( *(_BYTE *)a2 != 62 && *(_BYTE *)a2 != 61 || (v32 = *(_WORD *)(a2 + 2), ((v32 >> 7) & 6) == 0) && (v32 & 1) == 0 )
    {
      if ( (v31 & 1) == 0 )
        return 0;
      goto LABEL_6;
    }
  }
LABEL_37:
  v33 = sub_BD5C60(a2);
  v34 = *(_DWORD *)(a1 + 352);
  v35 = (_QWORD *)v33;
  v48 = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(a1 + 352) = v34 + 1;
  v9 = sub_BD2C40(88, unk_3F8E4CC);
  if ( !v9 )
    goto LABEL_27;
  v36 = sub_BCB120(v35);
  sub_BD35F0((__int64)v9, v36, 27);
  v37 = *((_DWORD *)v9 + 1);
  v9[4] = 0;
  v9[5] = 0;
  v9[9] = a2;
  v9[6] = 0;
  v12 = *(_BYTE *)v9 == 26;
  *((_DWORD *)v9 + 1) = v37 & 0x38000000 | 2;
  v9[7] = 0;
  v9[3] = sub_103AB40;
  v9[8] = v48;
  v38 = v9 - 8;
  if ( v12 )
    v38 = v9 - 4;
  if ( *v38 )
  {
    v39 = v38[1];
    *(_QWORD *)v38[2] = v39;
    if ( v39 )
      *(_QWORD *)(v39 + 16) = v38[2];
  }
  *v38 = 0;
  *((_DWORD *)v9 + 20) = v34;
  v22 = a1 + 32;
  *((_DWORD *)v9 + 21) = -1;
  v21 = *(_DWORD *)(a1 + 56);
  v49 = v7;
  if ( !v21 )
    goto LABEL_44;
LABEL_28:
  v23 = *(_QWORD *)(a1 + 40);
  v24 = 1;
  v25 = 0;
  v26 = (v21 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v27 = (__int64 *)(v23 + 16LL * v26);
  v28 = *v27;
  if ( v7 != *v27 )
  {
    while ( v28 != -4096 )
    {
      if ( !v25 && v28 == -8192 )
        v25 = v27;
      v26 = (v21 - 1) & (v24 + v26);
      v27 = (__int64 *)(v23 + 16LL * v26);
      v28 = *v27;
      if ( v7 == *v27 )
        goto LABEL_29;
      ++v24;
    }
    if ( !v25 )
      v25 = v27;
    v46 = *(_DWORD *)(a1 + 48);
    ++*(_QWORD *)(a1 + 32);
    v40 = v46 + 1;
    v50[0].m128i_i64[0] = (__int64)v25;
    if ( 4 * (v46 + 1) < 3 * v21 )
    {
      if ( v21 - *(_DWORD *)(a1 + 52) - v40 > v21 >> 3 )
        goto LABEL_83;
      goto LABEL_46;
    }
LABEL_45:
    v21 *= 2;
LABEL_46:
    sub_103FE70(v22, v21);
    sub_103F430(v22, &v49, v50);
    v7 = v49;
    v25 = (__int64 *)v50[0].m128i_i64[0];
    v40 = *(_DWORD *)(a1 + 48) + 1;
LABEL_83:
    *(_DWORD *)(a1 + 48) = v40;
    if ( *v25 != -4096 )
      --*(_DWORD *)(a1 + 52);
    *v25 = v7;
    v29 = v25 + 1;
    v25[1] = 0;
    goto LABEL_30;
  }
LABEL_29:
  v29 = v27 + 1;
LABEL_30:
  *v29 = v9;
  return v9;
}
