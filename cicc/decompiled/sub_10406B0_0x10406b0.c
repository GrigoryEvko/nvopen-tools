// Function: sub_10406B0
// Address: 0x10406b0
//
_QWORD *__fastcall sub_10406B0(__int64 a1, __int64 a2, __int64 *a3, _BYTE *a4)
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
  _QWORD *v31; // rdi
  char v32; // al
  unsigned __int16 v33; // dx
  __int64 v34; // rax
  int v35; // r14d
  _QWORD *v36; // r15
  __int64 v37; // rax
  int v38; // eax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  int v41; // edx
  __int64 v42; // rax
  unsigned int v43; // eax
  int v44; // esi
  __int64 v45; // rdx
  __int64 v46; // rdx
  int v47; // eax
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  __int64 v50; // [rsp+18h] [rbp-78h] BYREF
  __m128i v51[3]; // [rsp+20h] [rbp-70h] BYREF
  char v52; // [rsp+50h] [rbp-40h]

  v7 = a2;
  if ( *(_BYTE *)a2 == 85 )
  {
    v42 = *(_QWORD *)(a2 - 32);
    if ( v42 )
    {
      if ( !*(_BYTE *)v42 && *(_QWORD *)(v42 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v42 + 33) & 0x20) != 0 )
      {
        v43 = *(_DWORD *)(v42 + 36);
        if ( v43 == 155 )
          return 0;
        if ( v43 > 0x9B )
        {
          if ( v43 == 291 )
            return 0;
        }
        else if ( v43 > 6 )
        {
          if ( v43 == 11 )
            return 0;
        }
        else if ( v43 > 4 )
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
        v48 = *(_QWORD *)(a2 + 40);
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
          v9[8] = v48;
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
        if ( *(_BYTE *)a2 == 61 && (unsigned __int8)sub_103ADB0(a3, a2) )
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
              v44 = *(_DWORD *)(v17 + 80);
            else
              v44 = *(_DWORD *)(v17 + 72);
            *((_DWORD *)v9 + 20) = v44;
            if ( v15 )
            {
              v45 = *(v9 - 3);
              *(_QWORD *)*(v9 - 2) = v45;
              if ( v45 )
                *(_QWORD *)(v45 + 16) = *(v9 - 2);
            }
            *(v9 - 4) = v17;
            v46 = *(_QWORD *)(v17 + 16);
            *(v9 - 3) = v46;
            if ( v46 )
              *(_QWORD *)(v46 + 16) = v9 - 3;
            *(v9 - 2) = v17 + 16;
            *(_QWORD *)(v17 + 16) = v16;
          }
        }
LABEL_26:
        v21 = *(_DWORD *)(a1 + 56);
        v50 = v7;
        v22 = a1 + 32;
        if ( v21 )
          goto LABEL_27;
LABEL_43:
        ++*(_QWORD *)(a1 + 32);
        v51[0].m128i_i64[0] = 0;
        goto LABEL_44;
      }
      goto LABEL_36;
    }
    return 0;
  }
  v31 = (_QWORD *)*a3;
  v52 = 0;
  v32 = sub_CF63E0(v31, (unsigned __int8 *)a2, v51, (__int64)(a3 + 1));
  if ( (v32 & 2) == 0 )
  {
    if ( *(_BYTE *)a2 != 62 && *(_BYTE *)a2 != 61 || (v33 = *(_WORD *)(a2 + 2), ((v33 >> 7) & 6) == 0) && (v33 & 1) == 0 )
    {
      if ( (v32 & 1) == 0 )
        return 0;
      goto LABEL_6;
    }
  }
LABEL_36:
  v34 = sub_BD5C60(a2);
  v35 = *(_DWORD *)(a1 + 352);
  v36 = (_QWORD *)v34;
  v49 = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(a1 + 352) = v35 + 1;
  v9 = sub_BD2C40(88, unk_3F8E4CC);
  if ( !v9 )
    goto LABEL_26;
  v37 = sub_BCB120(v36);
  sub_BD35F0((__int64)v9, v37, 27);
  v38 = *((_DWORD *)v9 + 1);
  v9[4] = 0;
  v9[5] = 0;
  v9[9] = a2;
  v9[6] = 0;
  v12 = *(_BYTE *)v9 == 26;
  *((_DWORD *)v9 + 1) = v38 & 0x38000000 | 2;
  v9[7] = 0;
  v9[3] = sub_103AB40;
  v9[8] = v49;
  v39 = v9 - 8;
  if ( v12 )
    v39 = v9 - 4;
  if ( *v39 )
  {
    v40 = v39[1];
    *(_QWORD *)v39[2] = v40;
    if ( v40 )
      *(_QWORD *)(v40 + 16) = v39[2];
  }
  *v39 = 0;
  *((_DWORD *)v9 + 20) = v35;
  v22 = a1 + 32;
  *((_DWORD *)v9 + 21) = -1;
  v21 = *(_DWORD *)(a1 + 56);
  v50 = v7;
  if ( !v21 )
    goto LABEL_43;
LABEL_27:
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
        goto LABEL_28;
      ++v24;
    }
    if ( !v25 )
      v25 = v27;
    v47 = *(_DWORD *)(a1 + 48);
    ++*(_QWORD *)(a1 + 32);
    v41 = v47 + 1;
    v51[0].m128i_i64[0] = (__int64)v25;
    if ( 4 * (v47 + 1) < 3 * v21 )
    {
      if ( v21 - *(_DWORD *)(a1 + 52) - v41 > v21 >> 3 )
        goto LABEL_78;
      goto LABEL_45;
    }
LABEL_44:
    v21 *= 2;
LABEL_45:
    sub_103FE70(v22, v21);
    sub_103F430(v22, &v50, v51);
    v7 = v50;
    v25 = (__int64 *)v51[0].m128i_i64[0];
    v41 = *(_DWORD *)(a1 + 48) + 1;
LABEL_78:
    *(_DWORD *)(a1 + 48) = v41;
    if ( *v25 != -4096 )
      --*(_DWORD *)(a1 + 52);
    *v25 = v7;
    v29 = v25 + 1;
    v25[1] = 0;
    goto LABEL_29;
  }
LABEL_28:
  v29 = v27 + 1;
LABEL_29:
  *v29 = v9;
  return v9;
}
