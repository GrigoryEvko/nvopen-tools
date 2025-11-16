// Function: sub_395AC60
// Address: 0x395ac60
//
__int64 __fastcall sub_395AC60(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r12
  unsigned __int8 v4; // al
  __int64 v7; // rdx
  __int64 v8; // rdx
  unsigned int v9; // r8d
  unsigned __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 *v12; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // r8
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // r8
  int v25; // eax
  unsigned int *v26; // rcx
  __int64 *v27; // r10
  __int64 v28; // r12
  unsigned int *v29; // rbx
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned int v37; // esi
  int v38; // eax
  __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rax
  unsigned int v43; // esi
  int v44; // eax
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // [rsp+6D0h] [rbp-70h]
  __int64 *v48; // [rsp+6D8h] [rbp-68h]
  __int64 v49; // [rsp+6E0h] [rbp-60h]
  __int64 v50; // [rsp+6E8h] [rbp-58h]
  unsigned __int64 v51; // [rsp+6E8h] [rbp-58h]
  __int64 v52; // [rsp+6E8h] [rbp-58h]
  int v53; // [rsp+6E8h] [rbp-58h]
  __int64 v54; // [rsp+6F0h] [rbp-50h]
  __int64 v55; // [rsp+6F0h] [rbp-50h]
  __int64 v56; // [rsp+6F8h] [rbp-48h]
  __int64 v57; // [rsp+6F8h] [rbp-48h]
  unsigned int *v58; // [rsp+700h] [rbp-40h]
  int v59; // [rsp+700h] [rbp-40h]
  unsigned __int64 v60; // [rsp+700h] [rbp-40h]
  int v61; // [rsp+700h] [rbp-40h]
  __int64 *v62; // [rsp+708h] [rbp-38h]
  __int64 *v63; // [rsp+708h] [rbp-38h]
  __int64 v64; // [rsp+708h] [rbp-38h]
  __int64 *v65; // [rsp+708h] [rbp-38h]
  __int64 v66; // [rsp+708h] [rbp-38h]
  __int64 v67; // [rsp+708h] [rbp-38h]
  __int64 v68; // [rsp+708h] [rbp-38h]

  v3 = a2;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 <= 0x17u )
    return v3;
  while ( 1 )
  {
    if ( (unsigned __int8)(v4 - 61) <= 1u )
      goto LABEL_13;
    if ( v4 == 47 )
    {
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
        v14 = *(_QWORD *)(v3 - 8);
      else
        v14 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
      v15 = *(_QWORD *)(v14 + 24);
      if ( *(_BYTE *)(v15 + 16) != 13 )
        return v3;
      v16 = *(_DWORD *)(v15 + 32);
      v17 = *(__int64 **)(v15 + 24);
      v18 = v16 > 0x40 ? *v17 : (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
      v19 = v18 / 8;
      if ( *a3 < v19 )
        return v3;
      *a3 -= v19;
      if ( (*(_BYTE *)(v3 + 23) & 0x40) == 0 )
      {
LABEL_14:
        v12 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
LABEL_15:
        v3 = *v12;
        goto LABEL_16;
      }
LABEL_25:
      v12 = *(__int64 **)(v3 - 8);
      goto LABEL_15;
    }
    if ( (unsigned __int8)(v4 - 48) <= 1u )
    {
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
        v7 = *(_QWORD *)(v3 - 8);
      else
        v7 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
      v8 = *(_QWORD *)(v7 + 24);
      if ( *(_BYTE *)(v8 + 16) != 13 )
        return v3;
      v9 = *(_DWORD *)(v8 + 32);
      v10 = *(_QWORD *)(v8 + 24);
      if ( v4 == 49 )
      {
        if ( v9 <= 0x40 )
          v34 = (__int64)(v10 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
        else
          v34 = *(_QWORD *)v10;
        v11 = v34 / 8 + *a3;
      }
      else
      {
        if ( v9 > 0x40 )
          v10 = *(_QWORD *)v10;
        v11 = (v10 >> 3) + *a3;
      }
      *a3 = v11;
LABEL_13:
      if ( (*(_BYTE *)(v3 + 23) & 0x40) == 0 )
        goto LABEL_14;
      goto LABEL_25;
    }
    if ( v4 != 86 )
      break;
    v24 = **(_QWORD **)(v3 - 24);
    v56 = *(_QWORD *)(v3 - 24);
    v25 = *(unsigned __int8 *)(v24 + 8);
    if ( (unsigned int)(v25 - 13) > 1 )
      return v3;
    v26 = *(unsigned int **)(v3 + 56);
    v58 = &v26[*(unsigned int *)(v3 + 64)];
    if ( v26 != v58 )
    {
      v54 = v3;
      v27 = a3;
      v28 = a1;
      v29 = v26;
      v30 = v24;
      if ( (_BYTE)v25 == 13 )
      {
LABEL_37:
        v62 = v27;
        v31 = sub_15A9930(v28, v30);
        v27 = v62;
        *v62 += *(_QWORD *)(v31 + 8LL * *v29 + 16);
        v30 = *(_QWORD *)(*(_QWORD *)(v30 + 16) + 8LL * *v29);
        goto LABEL_38;
      }
      while ( 1 )
      {
        if ( (_BYTE)v25 != 14 )
          return v54;
        v30 = *(_QWORD *)(v30 + 24);
        v32 = 1;
        v33 = v30;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v33 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v39 = *(_QWORD *)(v33 + 32);
              v33 = *(_QWORD *)(v33 + 24);
              v32 *= v39;
              continue;
            case 1:
              v35 = 16;
              goto LABEL_49;
            case 2:
              v35 = 32;
              goto LABEL_49;
            case 3:
            case 9:
              v35 = 64;
              goto LABEL_49;
            case 4:
              v35 = 80;
              goto LABEL_49;
            case 5:
            case 6:
              v35 = 128;
              goto LABEL_49;
            case 7:
              v50 = v32;
              v37 = 0;
              v63 = v27;
              goto LABEL_56;
            case 0xB:
              v35 = *(_DWORD *)(v33 + 8) >> 8;
              goto LABEL_49;
            case 0xD:
              v52 = v32;
              v65 = v27;
              v41 = (_QWORD *)sub_15A9930(v28, v33);
              v27 = v65;
              v32 = v52;
              v35 = 8LL * *v41;
              goto LABEL_49;
            case 0xE:
              v47 = v32;
              v48 = v27;
              v49 = *(_QWORD *)(v33 + 24);
              v64 = *(_QWORD *)(v33 + 32);
              v51 = (unsigned int)sub_15A9FE0(v28, v49);
              v40 = sub_127FA20(v28, v49);
              v27 = v48;
              v32 = v47;
              v35 = 8 * v64 * v51 * ((v51 + ((unsigned __int64)(v40 + 7) >> 3) - 1) / v51);
              goto LABEL_49;
            case 0xF:
              v50 = v32;
              v63 = v27;
              v37 = *(_DWORD *)(v33 + 8) >> 8;
LABEL_56:
              v38 = sub_15A9520(v28, v37);
              v27 = v63;
              v32 = v50;
              v35 = (unsigned int)(8 * v38);
LABEL_49:
              *v27 += *v29 * ((unsigned __int64)(v35 * v32 + 7) >> 3);
              break;
          }
          break;
        }
LABEL_38:
        if ( ++v29 == v58 )
          break;
        LOBYTE(v25) = *(_BYTE *)(v30 + 8);
        if ( (_BYTE)v25 == 13 )
          goto LABEL_37;
      }
      a1 = v28;
      a3 = v27;
    }
    v3 = v56;
LABEL_16:
    v4 = *(_BYTE *)(v3 + 16);
    if ( v4 <= 0x17u )
      return v3;
  }
  if ( v4 == 83 )
  {
    v20 = *(_QWORD *)(v3 - 24);
    v21 = *(_QWORD **)(v20 + 24);
    v22 = *(_QWORD *)(**(_QWORD **)(v3 - 48) + 24LL);
    if ( *(_DWORD *)(v20 + 32) > 0x40u )
      v21 = (_QWORD *)*v21;
    v23 = 1;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v22 + 8) )
      {
        case 1:
          v36 = 16;
          goto LABEL_52;
        case 2:
          v36 = 32;
          goto LABEL_52;
        case 3:
        case 9:
          v36 = 64;
          goto LABEL_52;
        case 4:
          v36 = 80;
          goto LABEL_52;
        case 5:
        case 6:
          v36 = 128;
          goto LABEL_52;
        case 7:
          v59 = (int)v21;
          v43 = 0;
          v66 = v23;
          goto LABEL_66;
        case 0xB:
          v36 = *(_DWORD *)(v22 + 8) >> 8;
          goto LABEL_52;
        case 0xD:
          v61 = (int)v21;
          v68 = v23;
          v46 = (_QWORD *)sub_15A9930(a1, v22);
          v23 = v68;
          LODWORD(v21) = v61;
          v36 = 8LL * *v46;
          goto LABEL_52;
        case 0xE:
          v53 = (int)v21;
          v55 = v23;
          v57 = *(_QWORD *)(v22 + 24);
          v67 = *(_QWORD *)(v22 + 32);
          v60 = (unsigned int)sub_15A9FE0(a1, v57);
          v45 = sub_127FA20(a1, v57);
          v23 = v55;
          LODWORD(v21) = v53;
          v36 = 8 * v67 * v60 * ((v60 + ((unsigned __int64)(v45 + 7) >> 3) - 1) / v60);
          goto LABEL_52;
        case 0xF:
          v59 = (int)v21;
          v66 = v23;
          v43 = *(_DWORD *)(v22 + 8) >> 8;
LABEL_66:
          v44 = sub_15A9520(a1, v43);
          v23 = v66;
          LODWORD(v21) = v59;
          v36 = (unsigned int)(8 * v44);
LABEL_52:
          *a3 += (unsigned int)v21 * ((unsigned __int64)(v36 * v23 + 7) >> 3);
          v3 = *(_QWORD *)(v3 - 48);
          goto LABEL_16;
        case 0x10:
          v42 = *(_QWORD *)(v22 + 32);
          v22 = *(_QWORD *)(v22 + 24);
          v23 *= v42;
          continue;
        default:
          BUG();
      }
    }
  }
  return v3;
}
