// Function: sub_1932D10
// Address: 0x1932d10
//
__int64 __fastcall sub_1932D10(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edi
  unsigned int v8; // edx
  unsigned int *v9; // rax
  __int64 v10; // r10
  unsigned int v11; // r15d
  int v13; // eax
  __int64 v14; // r13
  unsigned int v15; // esi
  __int64 v16; // rcx
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 v19; // r10
  unsigned int v20; // esi
  __int64 v21; // rcx
  __int64 v22; // r9
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r10
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v29; // r9
  int v30; // eax
  unsigned int v31; // edx
  __int64 v32; // r10
  int v33; // ebx
  __int64 v34; // r11
  int v35; // ebx
  int v36; // ecx
  int v37; // ebx
  __int64 v38; // r11
  int v39; // ebx
  int v40; // edi
  int v41; // r11d
  __int64 v42; // r9
  int v43; // eax
  int v44; // edx
  __int64 v45; // rcx
  unsigned __int64 v46; // rax
  _QWORD *v47; // rbx
  int v48; // edx
  int v49; // eax
  unsigned __int64 v50; // rax
  char v51; // al
  __int64 v52; // rdx
  unsigned int v53; // esi
  int v54; // eax
  int v55; // eax
  __int64 v56; // rax
  int v57; // ebx
  __int64 v58; // r11
  int v59; // ebx
  int v60; // edx
  int v61; // r11d
  __int64 v62; // rbx
  __int64 v63; // [rsp+0h] [rbp-F0h]
  int v64; // [rsp+14h] [rbp-DCh]
  _QWORD *i; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v66; // [rsp+20h] [rbp-D0h]
  __int64 v67[2]; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v68; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v69; // [rsp+40h] [rbp-B0h] BYREF
  int v70; // [rsp+48h] [rbp-A8h]
  __int64 v71; // [rsp+B8h] [rbp-38h]

  v2 = a2;
  v4 = a2;
  v67[0] = a2;
  v5 = *(_DWORD *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  if ( !v5 )
  {
    v13 = *(unsigned __int8 *)(v4 + 16);
    if ( (unsigned __int8)v13 <= 0x17u )
    {
      ++*(_QWORD *)a1;
      goto LABEL_73;
    }
LABEL_7:
    switch ( v13 )
    {
      case 29:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 50:
      case 51:
      case 52:
      case 56:
      case 60:
      case 61:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
      case 68:
      case 69:
      case 70:
      case 71:
      case 75:
      case 76:
      case 78:
      case 79:
      case 83:
      case 84:
      case 85:
      case 87:
LABEL_8:
        v14 = sub_19335F0(a1, v4);
        if ( !v14 )
          goto LABEL_17;
        goto LABEL_9;
      case 54:
      case 55:
LABEL_15:
        if ( byte_42880A0[8 * ((*(unsigned __int16 *)(v4 + 18) >> 7) & 7) + 1] )
          goto LABEL_18;
        if ( sub_15F32D0(v4) )
        {
LABEL_17:
          v6 = *(_QWORD *)(a1 + 8);
          v5 = *(_DWORD *)(a1 + 24);
LABEL_18:
          if ( v5 )
          {
            v4 = v67[0];
            v7 = v5 - 1;
LABEL_20:
            v26 = v4;
            v27 = v7 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v28 = v6 + 16LL * v27;
            v29 = *(_QWORD *)v28;
            if ( v4 == *(_QWORD *)v28 )
              goto LABEL_21;
            v57 = 1;
            v58 = 0;
            while ( v29 != -8 )
            {
              if ( v29 == -16 && !v58 )
                v58 = v28;
              v27 = v7 & (v57 + v27);
              v28 = v6 + 16LL * v27;
              v29 = *(_QWORD *)v28;
              if ( v4 == *(_QWORD *)v28 )
                goto LABEL_21;
              ++v57;
            }
            v59 = *(_DWORD *)(a1 + 16);
            if ( v58 )
              v28 = v58;
            ++*(_QWORD *)a1;
            v60 = v59 + 1;
            if ( 4 * (v59 + 1) < 3 * v5 )
            {
              if ( v5 - (v60 + *(_DWORD *)(a1 + 20)) > v5 >> 3 )
              {
LABEL_81:
                *(_DWORD *)(a1 + 16) = v60;
                if ( *(_QWORD *)v28 != -8 )
                  --*(_DWORD *)(a1 + 20);
                *(_QWORD *)v28 = v26;
                *(_DWORD *)(v28 + 8) = 0;
                goto LABEL_21;
              }
LABEL_86:
              sub_177C7D0(a1, v5);
              sub_190E590(a1, v67, &v69);
              v28 = v69;
              v26 = v67[0];
              v60 = *(_DWORD *)(a1 + 16) + 1;
              goto LABEL_81;
            }
          }
          else
          {
            ++*(_QWORD *)a1;
          }
          v5 *= 2;
          goto LABEL_86;
        }
        v14 = sub_19335F0(a1, v4);
        *(_BYTE *)(v14 + 52) = *(_WORD *)(v4 + 18);
        *(_BYTE *)(v14 + 52) &= 1u;
LABEL_9:
        v15 = *(_DWORD *)(a1 + 56);
        v68 = v14;
        v63 = a1 + 32;
        if ( v15 )
        {
          v16 = *(_QWORD *)(a1 + 40);
          v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v18 = v16 + 16LL * v17;
          v19 = *(_QWORD *)v18;
          if ( *(_QWORD *)v18 == v14 )
          {
LABEL_11:
            v11 = *(_DWORD *)(v18 + 8);
            if ( v11 )
              goto LABEL_12;
LABEL_54:
            v69 = *(_QWORD *)(v14 + 40);
            LODWORD(v68) = *(_DWORD *)(v14 + 12);
            v46 = sub_1932650((int *)&v68, &v69, (int *)(v14 + 48), (char *)(v14 + 52));
            v47 = *(_QWORD **)(v14 + 24);
            v66 = v46;
            for ( i = &v47[*(unsigned int *)(v14 + 36)]; i != v47; v66 = sub_1593600(&v69, 0xCu, qword_4F99938) )
            {
              v48 = sub_1932D10(a1, *v47);
              if ( !byte_4F99930[0] )
              {
                v64 = v48;
                v49 = sub_2207590(byte_4F99930);
                v48 = v64;
                if ( v49 )
                {
                  v50 = unk_4FA04C8;
                  if ( !unk_4FA04C8 )
                    v50 = 0xFF51AFD7ED558CCDLL;
                  qword_4F99938 = v50;
                  sub_2207640(byte_4F99930);
                  v48 = v64;
                }
              }
              v70 = v48;
              ++v47;
              v71 = qword_4F99938;
              v69 = v66;
            }
            v68 = v66;
            if ( (unsigned __int8)sub_1932720(a1 + 64, &v68, &v69)
              && v69 != *(_QWORD *)(a1 + 72) + 16LL * *(unsigned int *)(a1 + 88) )
            {
              v11 = *(_DWORD *)(v69 + 8);
LABEL_12:
              v20 = *(_DWORD *)(a1 + 24);
              if ( v20 )
              {
                v21 = v67[0];
                v22 = *(_QWORD *)(a1 + 8);
                v23 = (v20 - 1) & ((LODWORD(v67[0]) >> 9) ^ (LODWORD(v67[0]) >> 4));
                v24 = v22 + 16LL * v23;
                v25 = *(_QWORD *)v24;
                if ( v67[0] == *(_QWORD *)v24 )
                {
LABEL_14:
                  *(_DWORD *)(v24 + 8) = v11;
                  return v11;
                }
                v37 = 1;
                v38 = 0;
                while ( v25 != -8 )
                {
                  if ( v25 == -16 && !v38 )
                    v38 = v24;
                  v23 = (v20 - 1) & (v37 + v23);
                  v24 = v22 + 16LL * v23;
                  v25 = *(_QWORD *)v24;
                  if ( v67[0] == *(_QWORD *)v24 )
                    goto LABEL_14;
                  ++v37;
                }
                v39 = *(_DWORD *)(a1 + 16);
                if ( v38 )
                  v24 = v38;
                ++*(_QWORD *)a1;
                v40 = v39 + 1;
                if ( 4 * (v39 + 1) < 3 * v20 )
                {
                  if ( v20 - *(_DWORD *)(a1 + 20) - v40 > v20 >> 3 )
                  {
LABEL_42:
                    *(_DWORD *)(a1 + 16) = v40;
                    if ( *(_QWORD *)v24 != -8 )
                      --*(_DWORD *)(a1 + 20);
                    *(_QWORD *)v24 = v21;
                    *(_DWORD *)(v24 + 8) = 0;
                    goto LABEL_14;
                  }
LABEL_89:
                  sub_177C7D0(a1, v20);
                  sub_190E590(a1, v67, &v69);
                  v24 = v69;
                  v21 = v67[0];
                  v40 = *(_DWORD *)(a1 + 16) + 1;
                  goto LABEL_42;
                }
              }
              else
              {
                ++*(_QWORD *)a1;
              }
              v20 *= 2;
              goto LABEL_89;
            }
            v11 = *(_DWORD *)(a1 + 280);
            *(_DWORD *)(a1 + 280) = v11 + 1;
            v68 = v66;
            v51 = sub_1932720(a1 + 64, &v68, &v69);
            v52 = v69;
            if ( v51 )
            {
LABEL_69:
              *(_DWORD *)(v52 + 8) = v11;
              v69 = v14;
              *((_DWORD *)sub_1932AD0(v63, &v69) + 2) = v11;
              goto LABEL_12;
            }
            v53 = *(_DWORD *)(a1 + 88);
            v54 = *(_DWORD *)(a1 + 80);
            ++*(_QWORD *)(a1 + 64);
            v55 = v54 + 1;
            if ( 4 * v55 >= 3 * v53 )
            {
              v53 *= 2;
            }
            else if ( v53 - *(_DWORD *)(a1 + 84) - v55 > v53 >> 3 )
            {
LABEL_66:
              *(_DWORD *)(a1 + 80) = v55;
              if ( *(_QWORD *)v52 != -1 )
                --*(_DWORD *)(a1 + 84);
              v56 = v68;
              *(_DWORD *)(v52 + 8) = 0;
              *(_QWORD *)v52 = v56;
              goto LABEL_69;
            }
            sub_1556300(a1 + 64, v53);
            sub_1932720(a1 + 64, &v68, &v69);
            v52 = v69;
            v55 = *(_DWORD *)(a1 + 80) + 1;
            goto LABEL_66;
          }
          v41 = 1;
          v42 = 0;
          while ( v19 != -8 )
          {
            if ( !v42 && v19 == -16 )
              v42 = v18;
            v17 = (v15 - 1) & (v41 + v17);
            v18 = v16 + 16LL * v17;
            v19 = *(_QWORD *)v18;
            if ( *(_QWORD *)v18 == v14 )
              goto LABEL_11;
            ++v41;
          }
          if ( !v42 )
            v42 = v18;
          v43 = *(_DWORD *)(a1 + 48);
          ++*(_QWORD *)(a1 + 32);
          v44 = v43 + 1;
          if ( 4 * (v43 + 1) < 3 * v15 )
          {
            v45 = v14;
            if ( v15 - *(_DWORD *)(a1 + 52) - v44 > v15 >> 3 )
            {
LABEL_51:
              *(_DWORD *)(a1 + 48) = v44;
              if ( *(_QWORD *)v42 != -8 )
                --*(_DWORD *)(a1 + 52);
              *(_QWORD *)v42 = v45;
              *(_DWORD *)(v42 + 8) = 0;
              goto LABEL_54;
            }
            v62 = a1 + 32;
            sub_1932910(v63, v15);
LABEL_94:
            sub_19327C0(v62, &v68, &v69);
            v42 = v69;
            v45 = v68;
            v44 = *(_DWORD *)(a1 + 48) + 1;
            goto LABEL_51;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 32);
        }
        v62 = a1 + 32;
        sub_1932910(v63, 2 * v15);
        goto LABEL_94;
      default:
        goto LABEL_18;
    }
  }
  v7 = v5 - 1;
  v8 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (unsigned int *)(v6 + 16LL * v8);
  v10 = *(_QWORD *)v9;
  if ( v4 != *(_QWORD *)v9 )
  {
    v30 = 1;
    while ( v10 != -8 )
    {
      v61 = v30 + 1;
      v8 = v7 & (v30 + v8);
      v9 = (unsigned int *)(v6 + 16LL * v8);
      v10 = *(_QWORD *)v9;
      if ( v4 == *(_QWORD *)v9 )
        goto LABEL_3;
      v30 = v61;
    }
    v13 = *(unsigned __int8 *)(v4 + 16);
    if ( (unsigned __int8)v13 <= 0x17u )
    {
LABEL_25:
      v31 = v7 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v28 = v6 + 16LL * v31;
      v32 = *(_QWORD *)v28;
      if ( v4 != *(_QWORD *)v28 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -8 )
        {
          if ( v32 == -16 && !v34 )
            v34 = v28;
          v31 = v7 & (v33 + v31);
          v28 = v6 + 16LL * v31;
          v32 = *(_QWORD *)v28;
          if ( v4 == *(_QWORD *)v28 )
            goto LABEL_21;
          ++v33;
        }
        v35 = *(_DWORD *)(a1 + 16);
        if ( v34 )
          v28 = v34;
        ++*(_QWORD *)a1;
        v36 = v35 + 1;
        if ( 4 * (v35 + 1) < 3 * v5 )
        {
          if ( v5 - *(_DWORD *)(a1 + 20) - v36 > v5 >> 3 )
          {
LABEL_32:
            *(_DWORD *)(a1 + 16) = v36;
            if ( *(_QWORD *)v28 != -8 )
              --*(_DWORD *)(a1 + 20);
            *(_QWORD *)v28 = v2;
            *(_DWORD *)(v28 + 8) = 0;
            goto LABEL_21;
          }
LABEL_74:
          sub_177C7D0(a1, v5);
          sub_190E590(a1, v67, &v69);
          v28 = v69;
          v2 = v67[0];
          v36 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_32;
        }
LABEL_73:
        v5 *= 2;
        goto LABEL_74;
      }
LABEL_21:
      v11 = *(_DWORD *)(a1 + 280);
      *(_DWORD *)(v28 + 8) = v11;
      *(_DWORD *)(a1 + 280) = v11 + 1;
      return v11;
    }
    goto LABEL_7;
  }
LABEL_3:
  if ( v9 == (unsigned int *)(v6 + 16LL * v5) )
  {
    if ( *(_BYTE *)(v4 + 16) > 0x17u )
    {
      switch ( *(_BYTE *)(v4 + 16) )
      {
        case 0x1D:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0x27:
        case 0x28:
        case 0x29:
        case 0x2A:
        case 0x2B:
        case 0x2C:
        case 0x2D:
        case 0x2E:
        case 0x2F:
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x38:
        case 0x3C:
        case 0x3D:
        case 0x3E:
        case 0x3F:
        case 0x40:
        case 0x41:
        case 0x42:
        case 0x43:
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x4B:
        case 0x4C:
        case 0x4E:
        case 0x4F:
        case 0x53:
        case 0x54:
        case 0x55:
        case 0x57:
          goto LABEL_8;
        case 0x36:
        case 0x37:
          goto LABEL_15;
        default:
          goto LABEL_20;
      }
    }
    goto LABEL_25;
  }
  return v9[2];
}
