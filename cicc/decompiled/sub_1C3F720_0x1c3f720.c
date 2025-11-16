// Function: sub_1C3F720
// Address: 0x1c3f720
//
float __fastcall sub_1C3F720(int a1, int a2, int a3, float a4, float a5)
{
  unsigned int v8; // esi
  unsigned int v9; // eax
  int v10; // r11d
  unsigned int v11; // edi
  unsigned int v12; // edx
  float v13; // r10d
  int v14; // eax
  unsigned int v15; // edi
  unsigned int v16; // esi
  char v17; // cl
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rbx
  __int64 v21; // rax
  int v22; // r12d
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  unsigned int v26; // edi
  int v27; // esi
  int v28; // r12d
  int v29; // ebx
  int v30; // esi
  float result; // xmm0_4
  unsigned __int64 v32; // rax
  int v33; // esi
  float v34; // xmm1_4
  float v35; // xmm2_4
  unsigned int v36; // eax
  unsigned int v37; // edi
  int v38; // r12d
  bool v39; // sf
  float v40; // eax
  int v41; // eax
  int v42; // esi
  float v43; // xmm1_4
  float v44; // xmm2_4
  unsigned int v45; // eax
  int v46; // eax
  int v47; // esi
  unsigned int v48; // eax
  __int64 v49; // rsi
  __int64 v50; // rcx
  __int64 v51; // rbx
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // r8
  __int64 v55; // rbx
  signed __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rcx
  float v59; // eax
  int v60; // eax
  unsigned int v61; // eax
  int v62; // edi
  int v63; // esi

  v8 = LODWORD(a4);
  v9 = LODWORD(a5);
  if ( a1 )
  {
    if ( 2 * LODWORD(a4) < (unsigned int)&loc_1000000 )
      v8 = LODWORD(a4) & 0x80000000;
    if ( 2 * LODWORD(a5) < (unsigned int)&loc_1000000 )
      v9 = LODWORD(a5) & 0x80000000;
  }
  v10 = v9 ^ v8;
  v11 = (unsigned __int8)(v8 >> 23) - 1;
  v12 = (unsigned __int8)(v9 >> 23) - 1;
  LODWORD(v13) = (v9 ^ v8) & 0x80000000;
  if ( v11 <= 0xFD && v12 <= 0xFD )
    goto LABEL_8;
  v29 = 2 * v8;
  if ( 2 * v8 <= 0xFF000000 )
  {
    v38 = 2 * v9;
    if ( 2 * v9 > 0xFF000000 )
    {
      v46 = v9 | 0x400000;
      if ( a2 )
        *(float *)&v46 = NAN;
      return *(float *)&v46;
    }
    else if ( !(v38 | v29) || v38 == -16777216 && v29 == -16777216 )
    {
      LODWORD(result) = a2 == 0 ? -4194304 : 0x7FFFFFFF;
    }
    else if ( !v29 || v38 == -16777216 )
    {
      LODWORD(result) = (v9 ^ v8) & 0x80000000;
    }
    else
    {
      if ( v38 && v29 != -16777216 )
      {
        if ( !(unsigned __int8)(v8 >> 23) )
        {
          v63 = v8 << 9;
          if ( v63 >= 0 )
          {
            do
            {
              --v11;
              v39 = (v63 & 0x40000000) != 0;
              v63 *= 2;
            }
            while ( !v39 );
          }
          v8 = (unsigned int)v63 >> 8;
        }
        if ( !(unsigned __int8)(v9 >> 23) )
        {
          v14 = v9 << 9;
          if ( v14 >= 0 )
          {
            do
            {
              --v12;
              v39 = (v14 & 0x40000000) != 0;
              v14 *= 2;
            }
            while ( !v39 );
          }
LABEL_9:
          v15 = v11 - v12;
          v16 = v8 & 0x7FFFFF | 0x800000;
          v17 = v15 + 126;
          v18 = v14 | 0x80000000;
          v19 = byte_42D0EC0[BYTE3(v18) - 128];
          v20 = v16;
          v21 = (unsigned int)(2 * ((v19 << 24) - ((v18 * (unsigned __int64)(unsigned int)((v19 * v19) << 16)) >> 32)));
          v22 = v15 + 126;
          v23 = v16
              * ((((unsigned __int64)(unsigned int)-((unsigned __int64)(v21 * v18) >> 32) * v21) >> 31) & 0xFFFFFFFE);
          if ( (int)(HIDWORD(v23) << 8) > 0 )
          {
            v22 = v15 + 125;
            v23 *= 2LL;
          }
          v24 = (unsigned int)v18 >> 8;
          if ( !a3 )
          {
            v32 = HIDWORD(v23);
            v26 = v32;
            if ( (unsigned int)v22 <= 0xFD )
            {
              v33 = (v16 << (((v32 * (unsigned int)v24) >> 47) + 23)) - v32 * v24;
              v34 = (float)(v33 - v24);
              if ( v34 < 0.0 )
                v34 = 0.0 - v34;
              v35 = (float)v33;
              if ( (float)v33 < 0.0 )
                v35 = 0.0 - v35;
              v36 = v32 + 1;
              if ( v35 > v34 )
                v26 = v36;
              goto LABEL_31;
            }
            if ( v22 > 253 )
              goto LABEL_77;
            if ( !a1 )
            {
LABEL_66:
              if ( v16 == (_DWORD)v24 && v22 >= -24 )
              {
                LODWORD(v25) = 0x800000 >> -v17;
                if ( a3 == 3 )
                {
                  if ( !(_DWORD)v25 )
                  {
                    LODWORD(v13) |= v10 >= 0;
                    return v13;
                  }
                }
                else if ( a3 == 2 && !(_DWORD)v25 )
                {
                  LODWORD(v13) |= (unsigned int)v10 >> 31;
                  return v13;
                }
                goto LABEL_71;
              }
              if ( v22 < -23 )
              {
                if ( a3 )
                {
                  if ( a3 != 1 )
                  {
                    if ( a3 == 3 )
                    {
                      LODWORD(v13) |= v10 >= 0;
                    }
                    else
                    {
                      v62 = LODWORD(v13) | v26;
                      LODWORD(v13) |= (unsigned int)v10 >> 31;
                      if ( a3 != 2 )
                        return *(float *)&v62;
                    }
                  }
                  return v13;
                }
                LODWORD(v25) = v22 == -24;
                goto LABEL_71;
              }
              v37 = v26 >> -(char)v22;
              v49 = v24 * v37;
              v50 = v20 << ((unsigned __int8)v22 + 23);
              v51 = 2 * v50;
              v52 = v50 - v49;
              v53 = 2 * v52 - v24;
              v54 = v53 + v51;
              v55 = v51 - v49;
              if ( (__int64)abs64(v53) > (__int64)abs64(v54) )
                v52 = v55;
              switch ( a3 )
              {
                case 0:
                  v56 = abs64(v52);
                  v57 = v24 - v52;
                  v58 = v52 - v24;
                  if ( v58 < 0 )
                    v58 = v57;
                  if ( v56 <= v58 && (v56 != v58 || (v37 & 1) == 0) )
                    goto LABEL_32;
LABEL_102:
                  ++v37;
                  goto LABEL_32;
                case 1:
                  if ( v52 >= 0 || !v37 )
                    goto LABEL_32;
                  break;
                case 3:
                  if ( v52 >= 0 || !v37 )
                  {
                    if ( v52 <= 0 || v10 < 0 )
                      goto LABEL_32;
                    goto LABEL_102;
                  }
                  if ( v10 >= 0 )
                    goto LABEL_32;
                  break;
                case 2:
                  if ( v52 >= 0 || !v37 )
                  {
                    if ( v52 <= 0 || v10 >= 0 )
                      goto LABEL_32;
                    goto LABEL_102;
                  }
                  if ( v10 < 0 )
                    goto LABEL_32;
                  break;
                default:
                  goto LABEL_32;
              }
              LODWORD(v13) |= v37 - 1;
              return v13;
            }
            v42 = (v16 << (((v32 * (unsigned int)v24) >> 47) + 23)) - v32 * v24;
            v43 = (float)(v42 - v24);
            if ( v43 < 0.0 )
              v43 = 0.0 - v43;
            v44 = (float)v42;
            if ( (float)v42 < 0.0 )
              v44 = 0.0 - v44;
            v45 = v32 + 1;
            if ( v44 > v43 )
              v26 = v45;
            goto LABEL_82;
          }
          v25 = (v23 + 0x80000000) >> 32;
          v26 = v25;
          if ( (unsigned int)v22 <= 0xFD )
          {
            v27 = (v16 << (((v25 * v24) >> 47) + 23)) - v25 * v24;
            switch ( a3 )
            {
              case 1:
                v48 = v25 - 1;
                if ( v27 < 0 )
                  v26 = v48;
                v37 = (v22 << 23) + v26;
                if ( v37 == 2139095040 )
                  goto LABEL_19;
                goto LABEL_32;
              case 3:
                v28 = v22 << 23;
                if ( v27 < 0 && v10 < 0 )
                {
LABEL_18:
                  LODWORD(v25) = v28 + v25 - 1;
                  if ( (_DWORD)v25 == 2139095040 )
                  {
LABEL_19:
                    LODWORD(v13) |= 0x7F7FFFFFu;
                    return v13;
                  }
LABEL_71:
                  LODWORD(v13) |= v25;
                  return v13;
                }
                if ( v27 <= 0 || v10 < 0 )
                {
                  LODWORD(v25) = v28 + v25;
                  if ( (_DWORD)v25 != 2139095040 )
                    goto LABEL_71;
                  if ( v10 < 0 )
                    goto LABEL_19;
                  goto LABEL_77;
                }
                goto LABEL_76;
              case 2:
                v28 = v22 << 23;
                if ( v27 < 0 && v10 >= 0 )
                  goto LABEL_18;
                if ( v27 <= 0 || v10 >= 0 )
                {
                  LODWORD(v25) = v28 + v25;
                  if ( (_DWORD)v25 != 2139095040 )
                    goto LABEL_71;
                  if ( v10 >= 0 )
                    goto LABEL_19;
                  goto LABEL_77;
                }
LABEL_76:
                LODWORD(v25) = v28 + v25 + 1;
                if ( (_DWORD)v25 != 2139095040 )
                  goto LABEL_71;
LABEL_77:
                LODWORD(v13) |= 0x7F800000u;
                return v13;
            }
LABEL_31:
            v37 = (v22 << 23) + v26;
            if ( v37 == 2139095040 )
              goto LABEL_77;
LABEL_32:
            LODWORD(v13) |= v37;
            return v13;
          }
          if ( v22 > 253 )
          {
            switch ( a3 )
            {
              case 1:
                goto LABEL_19;
              case 3:
                v59 = v13;
                LODWORD(v13) |= 0x7F800000u;
                v60 = LODWORD(v59) | 0x7F7FFFFF;
                if ( v10 < 0 )
                  return *(float *)&v60;
                return v13;
              case 2:
                v40 = v13;
                LODWORD(v13) |= 0x7F7FFFFFu;
                v41 = LODWORD(v40) | 0x7F800000;
                if ( v10 < 0 )
                  return *(float *)&v41;
                return v13;
            }
            goto LABEL_71;
          }
          if ( !a1 )
            goto LABEL_66;
          v47 = (v16 << (((v25 * v24) >> 47) + 23)) - v25 * v24;
          switch ( a3 )
          {
            case 1:
              v61 = v25 - 1;
              if ( v47 < 0 )
                v26 = v61;
              goto LABEL_82;
            case 3:
              if ( v47 >= 0 || v10 >= 0 )
              {
                if ( v47 > 0 && v10 >= 0 )
                  goto LABEL_135;
                goto LABEL_82;
              }
              break;
            case 2:
              if ( v47 >= 0 || v10 < 0 )
              {
                if ( v47 > 0 && v10 < 0 )
LABEL_135:
                  v26 = v25 + 1;
LABEL_82:
                if ( (v22 << 23) + v26 == 0x800000 )
                  LODWORD(v13) |= 0x800000u;
                return v13;
              }
              break;
            default:
              goto LABEL_82;
          }
          v26 = v25 - 1;
          goto LABEL_82;
        }
LABEL_8:
        v14 = v9 << 8;
        goto LABEL_9;
      }
      LODWORD(result) = LODWORD(v13) | 0x7F800000;
    }
  }
  else
  {
    v30 = v8 | 0x400000;
    if ( a2 )
      *(float *)&v30 = NAN;
    return *(float *)&v30;
  }
  return result;
}
