// Function: sub_30776F0
// Address: 0x30776f0
//
__int64 __fastcall sub_30776F0(__int64 a1, __int64 a2, __int64 a3, int *a4, __int64 *a5)
{
  __int64 v7; // r13
  _QWORD *v8; // r15
  _QWORD *v9; // r14
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r13
  __int64 v23; // r14
  char v24; // al
  __int64 v25; // rbx
  __int64 v26; // r12
  char v27; // al
  unsigned __int8 *v28; // r13
  __int64 v29; // rax
  unsigned __int8 *v30; // rbx
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 result; // rax
  __int64 v34; // rdi
  unsigned __int8 **v35; // r13
  __int64 v36; // rax
  int v37; // esi
  __int64 v38; // r14
  unsigned __int8 *v39; // r12
  unsigned __int8 **v40; // r15
  __int64 v41; // rbx
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // r13
  __int64 v44; // r14
  unsigned __int8 v45; // dl
  unsigned __int64 v46; // r13
  __int64 v47; // rax
  unsigned __int64 v48; // rax
  unsigned int v49; // eax
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+20h] [rbp-60h]
  unsigned int v54; // [rsp+28h] [rbp-58h]
  unsigned int v55; // [rsp+2Ch] [rbp-54h]
  __int64 v56; // [rsp+30h] [rbp-50h]
  __int64 v57; // [rsp+30h] [rbp-50h]
  __int64 v58; // [rsp+38h] [rbp-48h]
  int v59; // [rsp+38h] [rbp-48h]
  __int64 v60[7]; // [rsp+48h] [rbp-38h] BYREF

  v60[0] = a2;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = sub_C52410();
  v9 = v8 + 1;
  v10 = sub_C959E0();
  v11 = (_QWORD *)v8[2];
  if ( v11 )
  {
    v12 = v8 + 1;
    do
    {
      while ( 1 )
      {
        v13 = v11[2];
        v14 = v11[3];
        if ( v10 <= v11[4] )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v14 )
          goto LABEL_6;
      }
      v12 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v13 );
LABEL_6:
    if ( v12 != v9 && v10 >= v12[4] )
      v9 = v12;
  }
  if ( v9 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_93;
  v15 = v9[7];
  if ( !v15 )
    goto LABEL_93;
  v16 = v9 + 6;
  v10 = dword_503A828;
  do
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v15 + 16);
      v18 = *(_QWORD *)(v15 + 24);
      if ( *(_DWORD *)(v15 + 32) >= (signed int)dword_503A828 )
        break;
      v15 = *(_QWORD *)(v15 + 24);
      if ( !v18 )
        goto LABEL_15;
    }
    v16 = (_QWORD *)v15;
    v15 = *(_QWORD *)(v15 + 16);
  }
  while ( v17 );
LABEL_15:
  if ( v16 == v9 + 6 || (signed int)dword_503A828 < *((_DWORD *)v16 + 8) || *((int *)v16 + 9) <= 0 )
  {
LABEL_93:
    v19 = *(_DWORD *)(*(_QWORD *)(v7 + 200) + 8LL);
    if ( !v19 )
      goto LABEL_27;
  }
  else
  {
    v19 = qword_503A868[8];
  }
  v20 = *(_QWORD *)(v60[0] + 40);
  v21 = *(_QWORD *)(v60[0] + 32);
  if ( v21 == v20 )
  {
LABEL_26:
    a4[3] = v19;
    a4[2] = 0;
    a4[4] = 0;
    a4[10] = 2;
    goto LABEL_27;
  }
  while ( 1 )
  {
    v22 = *(_QWORD *)(*(_QWORD *)v21 + 56LL);
    v23 = *(_QWORD *)v21 + 48LL;
    if ( v22 != v23 )
      break;
LABEL_25:
    v21 += 8;
    if ( v20 == v21 )
      goto LABEL_26;
  }
  while ( 1 )
  {
    if ( !v22 )
      BUG();
    v24 = *(_BYTE *)(v22 - 24);
    if ( v24 != 34 && v24 != 85 )
      goto LABEL_24;
    v34 = *(_QWORD *)(v22 - 56);
    if ( !v34 )
      break;
    if ( *(_BYTE *)v34 )
      break;
    if ( *(_QWORD *)(v34 + 24) != *(_QWORD *)(v22 + 56) )
      break;
    v57 = v20;
    v59 = v19;
    if ( (unsigned __int8)sub_3071FB0((_BYTE *)v34) )
      break;
    v19 = v59;
    v20 = v57;
LABEL_24:
    v22 = *(_QWORD *)(v22 + 8);
    if ( v23 == v22 )
      goto LABEL_25;
  }
  if ( a5 )
  {
    v10 = (unsigned __int64)v60;
    sub_3077210(a5, v60, (unsigned __int8 *)(v22 - 24));
  }
LABEL_27:
  *((_WORD *)a4 + 22) = 257;
  *((_BYTE *)a4 + 49) = 1;
  v51 = *(_QWORD *)(a2 + 40);
  if ( *(_QWORD *)(a2 + 32) != v51 )
  {
    v58 = *(_QWORD *)(a2 + 32);
    v54 = 1;
    while ( 1 )
    {
      v25 = *(_QWORD *)(*(_QWORD *)v58 + 56LL);
      v26 = *(_QWORD *)v58 + 48LL;
      if ( v25 != v26 )
        break;
LABEL_44:
      v58 += 8;
      if ( v51 == v58 )
        goto LABEL_45;
    }
    while ( 1 )
    {
LABEL_33:
      if ( !v25 )
        BUG();
      v27 = *(_BYTE *)(v25 - 24);
      if ( v27 == 62 || v27 == 61 )
      {
        v28 = sub_BD3990(*(unsigned __int8 **)(v25 - 56), v10);
        v29 = *((_QWORD *)v28 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 <= 1 )
          v29 = **(_QWORD **)(v29 + 16);
        if ( *(_DWORD *)(v29 + 8) >> 8 == 5 && *v28 == 63 )
          break;
      }
      v25 = *(_QWORD *)(v25 + 8);
      if ( v26 == v25 )
        goto LABEL_44;
    }
    v55 = 1;
    v53 = v26;
    v52 = v25;
    v30 = v28;
    while ( 2 )
    {
      if ( (unsigned __int8)sub_B4DD90((__int64)v30) )
      {
        v31 = *((_DWORD *)v30 + 1) & 0x7FFFFFF;
        goto LABEL_42;
      }
      v35 = (unsigned __int8 **)&v30[32 * (1LL - (*((_DWORD *)v30 + 1) & 0x7FFFFFF))];
      if ( **v35 > 0x15u )
        v55 *= (_DWORD)qword_502D1A8;
      if ( (v30[7] & 0x40) != 0 )
        v35 = (unsigned __int8 **)(*((_QWORD *)v30 - 1) + 32LL);
      v36 = sub_BB5290((__int64)v30);
      v37 = *((_DWORD *)v30 + 1);
      v31 = v37 & 0x7FFFFFF;
      v38 = v36 & 0xFFFFFFFFFFFFFFF9LL | 4;
      if ( (_DWORD)v31 == 2 )
      {
LABEL_86:
        v49 = v54;
        v10 = v55;
        if ( v54 < v55 )
          v49 = v55;
        v54 = v49;
LABEL_42:
        v30 = sub_BD3990(*(unsigned __int8 **)&v30[-32 * v31], v10);
        if ( *v30 != 63 )
        {
          v26 = v53;
          v25 = *(_QWORD *)(v52 + 8);
          if ( v53 == v25 )
            goto LABEL_44;
          goto LABEL_33;
        }
        continue;
      }
      break;
    }
    v56 = (unsigned int)(v31 - 3);
    v39 = v30;
    v40 = v35;
    v41 = 0;
    while ( 2 )
    {
      v42 = v38 & 0xFFFFFFFFFFFFFFF8LL;
      v43 = v38 & 0xFFFFFFFFFFFFFFF8LL;
      if ( **(_BYTE **)&v39[32 * ((unsigned int)(v41 + 2) - v31)] <= 0x15u )
      {
LABEL_61:
        if ( v38 )
          goto LABEL_62;
LABEL_64:
        v43 = sub_BCBAE0(v42, *v40, v31);
LABEL_65:
        v37 = *((_DWORD *)v39 + 1);
LABEL_66:
        v45 = *(_BYTE *)(v43 + 8);
        if ( v45 == 16 )
        {
          v38 = *(_QWORD *)(v43 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
        }
        else
        {
          v46 = v43 & 0xFFFFFFFFFFFFFFF9LL;
          if ( (unsigned int)v45 - 17 > 1 )
          {
            v38 = 0;
            if ( v45 == 15 )
              v38 = v46;
          }
          else
          {
            v38 = v46 | 2;
          }
        }
        v40 += 4;
        v31 = v37 & 0x7FFFFFF;
        if ( v41 == v56 )
        {
          v30 = v39;
          goto LABEL_86;
        }
        ++v41;
        continue;
      }
      break;
    }
    if ( !v38 )
    {
      v48 = sub_BCBAE0(v42, *v40, v31);
      if ( *(_BYTE *)(v48 + 8) != 16 )
        goto LABEL_64;
      goto LABEL_77;
    }
    v47 = (v38 >> 1) & 3;
    if ( v47 == 2 )
    {
      if ( v42 )
      {
        v48 = v38 & 0xFFFFFFFFFFFFFFF8LL;
        if ( *(_BYTE *)(v42 + 8) != 16 )
          goto LABEL_66;
LABEL_77:
        v55 *= *(_DWORD *)(v48 + 32);
        goto LABEL_61;
      }
LABEL_89:
      v48 = sub_BCBAE0(v42, *v40, v31);
      v42 = v38 & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v48 + 8) == 16 )
        goto LABEL_77;
LABEL_62:
      v44 = (v38 >> 1) & 3;
      if ( v44 == 2 )
      {
        if ( !v42 )
          goto LABEL_64;
        goto LABEL_65;
      }
      if ( v44 != 1 || !v42 )
        goto LABEL_64;
      v37 = *((_DWORD *)v39 + 1);
      v31 = *(_QWORD *)(v42 + 24);
    }
    else
    {
      if ( v47 != 1 || !v42 )
        goto LABEL_89;
      v31 = *(_QWORD *)(v42 + 24);
      v48 = v31;
      if ( *(_BYTE *)(v31 + 8) == 16 )
        goto LABEL_77;
    }
    v43 = v31;
    goto LABEL_66;
  }
  v54 = 1;
LABEL_45:
  a4[16] = v54;
  v32 = *a4;
  *((_BYTE *)a4 + 69) = 1;
  result = v32 >> 1;
  a4[3] = result;
  return result;
}
