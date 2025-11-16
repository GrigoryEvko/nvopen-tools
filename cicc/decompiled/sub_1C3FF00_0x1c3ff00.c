// Function: sub_1C3FF00
// Address: 0x1c3ff00
//
float __fastcall sub_1C3FF00(int a1, int a2, int a3, float a4, float a5, float a6)
{
  unsigned int v9; // edi
  unsigned int v10; // esi
  unsigned int v11; // r9d
  unsigned int v12; // edx
  unsigned int v13; // eax
  int v14; // ecx
  int v15; // ebx
  float result; // xmm0_4
  int v17; // r12d
  unsigned int v18; // eax
  float v19; // edx
  __int64 v20; // rsi
  unsigned int v21; // r10d
  unsigned int v22; // edi
  int v23; // eax
  unsigned int v24; // r12d
  unsigned int v25; // esi
  unsigned int v26; // r9d
  int v27; // ebx
  unsigned int v28; // r10d
  float v29; // ebx
  int v30; // esi
  unsigned int v31; // ebx
  unsigned int v32; // r13d
  unsigned int v33; // kr00_4
  unsigned int v34; // esi
  int v35; // edi
  unsigned int v36; // edx
  int v37; // r13d
  unsigned int v38; // esi
  unsigned int v39; // eax
  float v40; // esi
  unsigned int v41; // eax
  unsigned int v42; // r14d
  int v43; // ecx
  int v44; // ebx
  int v45; // esi
  unsigned int v46; // ecx
  unsigned int v47; // edi
  unsigned int v48; // edi
  bool v49; // r8
  int v50; // eax
  unsigned int v51; // eax
  unsigned int v52; // r9d
  bool v53; // sf
  unsigned int v54; // r12d
  bool v55; // cl
  bool v56; // al
  bool v57; // dl
  unsigned int v58; // esi
  unsigned int v59; // edi
  __int16 v60; // [rsp+1h] [rbp-ADh]

  if ( a1 )
  {
    if ( (unsigned int)(2 * LODWORD(a4)) <= 0xFFFFFF )
      LODWORD(a4) &= 0x80000000;
    if ( (unsigned int)(2 * LODWORD(a5)) <= 0xFFFFFF )
      LODWORD(a5) &= 0x80000000;
    if ( (unsigned int)(2 * LODWORD(a6)) <= 0xFFFFFF )
      LODWORD(a6) &= 0x80000000;
  }
  v9 = LODWORD(a4);
  v10 = LODWORD(a5);
  v11 = LODWORD(a6);
  v12 = (unsigned __int8)(LODWORD(a4) >> 23) - 1;
  v13 = (unsigned __int8)(LODWORD(a5) >> 23) - 1;
  v14 = (unsigned __int8)(LODWORD(a6) >> 23) - 1;
  if ( v13 <= 0xFD && v12 <= 0xFD && (unsigned int)v14 <= 0xFD )
  {
    v17 = 2 * LODWORD(a6);
    goto LABEL_11;
  }
  v15 = 2 * LODWORD(a5);
  if ( (unsigned int)(2 * LODWORD(a5)) > 0xFF000000 )
  {
    if ( a2 )
      return NAN;
    else
      return a5 + a5;
  }
  v17 = 2 * LODWORD(a6);
  if ( (unsigned int)(2 * LODWORD(a6)) > 0xFF000000 )
  {
    if ( a2 )
      return NAN;
    else
      return a6 + a6;
  }
  v37 = 2 * LODWORD(a4);
  if ( (unsigned int)(2 * LODWORD(a4)) > 0xFF000000 )
  {
    if ( a2 )
      return NAN;
    else
      return a4 + a4;
  }
  LOBYTE(v60) = v37 == 0;
  if ( !v37 && v15 == -16777216 || (HIBYTE(v60) = v15 == 0, v37 == -16777216) && !v15 )
  {
    LODWORD(result) = a2 == 0 ? -4194304 : 0x7FFFFFFF;
    return result;
  }
  if ( v17 == -16777216 )
  {
    if ( v37 != -16777216 && v15 != -16777216 )
      return a6;
    if ( (LODWORD(a6) ^ LODWORD(a4) ^ LODWORD(a5)) < 0 )
    {
      LODWORD(result) = a2 == 0 ? -4194304 : 0x7FFFFFFF;
      return result;
    }
  }
  if ( v37 == -16777216 )
  {
    LODWORD(result) = LODWORD(a5) & 0x80000000 ^ LODWORD(a4);
    return result;
  }
  if ( v15 == -16777216 )
  {
    LODWORD(result) = LODWORD(a4) & 0x80000000 ^ LODWORD(a5);
    return result;
  }
  if ( v17 == -16777216 )
    return a6;
  if ( LODWORD(a6) == 0x80000000 )
  {
    if ( !v60 )
      goto LABEL_128;
    if ( (LODWORD(a4) ^ LODWORD(a5)) < 0 )
      return -0.0;
  }
  else
  {
    if ( v17 )
    {
      if ( v60 )
        return a6;
      goto LABEL_128;
    }
    if ( !v60 )
    {
LABEL_128:
      if ( !(unsigned __int8)(LODWORD(a4) >> 23) )
      {
        v59 = LODWORD(a4) << 8;
        if ( (LODWORD(a4) & 0x800000) == 0 )
        {
          do
          {
            --v12;
            v53 = (v59 & 0x40000000) != 0;
            v59 *= 2;
          }
          while ( !v53 );
        }
        ++v12;
        v9 = LODWORD(a4) & 0x80000000 | (v59 >> 8);
      }
      if ( !(unsigned __int8)(LODWORD(a5) >> 23) )
      {
        v58 = LODWORD(a5) << 8;
        if ( (LODWORD(a5) & 0x800000) == 0 )
        {
          do
          {
            --v13;
            v53 = (v58 & 0x40000000) != 0;
            v58 *= 2;
          }
          while ( !v53 );
        }
        ++v13;
        v10 = LODWORD(a5) & 0x80000000 | (v58 >> 8);
      }
      if ( !(unsigned __int8)(LODWORD(a6) >> 23) && v17 )
      {
        v52 = LODWORD(a6) << 8;
        if ( (LODWORD(a6) & 0x800000) == 0 )
        {
          do
          {
            --v14;
            v53 = (v52 & 0x40000000) != 0;
            v52 *= 2;
          }
          while ( !v53 );
        }
        ++v14;
        v54 = v52 >> 8;
        v11 = (v52 >> 8) | LODWORD(a6) & 0x80000000;
        v17 = 2 * v54;
      }
LABEL_11:
      v18 = v12 + v13;
      LODWORD(v19) = (v10 ^ v9) & 0x80000000;
      v20 = (v9 & 0xFFFFFF | 0x800000LL) * ((v10 << 8) | 0x80000000);
      v21 = v20;
      v22 = HIDWORD(v20);
      if ( HIDWORD(v20) <= 0x7FFFFF )
      {
        v21 = 2 * v20;
        v23 = v18 - 126;
        v22 = v20 >> 31;
      }
      else
      {
        v23 = v18 - 125;
      }
      if ( v17 )
      {
        v24 = 0;
        v25 = v11 & 0x80000000;
        v26 = v11 & 0x7FFFFF | 0x800000;
        if ( v14 > v23 )
        {
          v27 = v23;
          v24 = v21;
          v23 = v14;
          v28 = v22;
          v14 = v27;
          v29 = *(float *)&v25;
          v22 = v26;
          v25 = LODWORD(v19);
          v26 = v28;
          v19 = v29;
          v21 = 0;
        }
        v30 = LODWORD(v19) ^ v25;
        v31 = v23 - v14;
        if ( (unsigned int)(v23 - v14) > 0x31 )
        {
          v32 = 1;
          v24 = 0;
          v26 = 0;
        }
        else
        {
          v32 = 0;
          if ( v31 > 0x1F )
          {
            v32 = v24;
            v31 -= 32;
            v24 = v26;
            v26 = 0;
          }
          if ( v31 )
          {
            v32 = (v24 << (32 - v31)) | (v32 >> v31) | (v32 << (32 - v31) != 0);
            v24 = (v26 << (32 - v31)) | (v24 >> v31);
            v26 >>= v31;
          }
        }
        if ( v30 < 0 )
        {
          v42 = v32 != 0;
          v43 = (__PAIR64__(v22 - v26, v21 - v42) - __PAIR64__(v21 < v42, v24)) >> 32;
          v44 = v21 - v42 - v24;
          if ( !(v43 | v44 | -v32) )
          {
            if ( a3 == 2 )
              return -0.0;
            else
              return 0.0;
          }
          if ( v43 < 0 )
          {
            v43 = ~v43;
            v44 = ~v44;
            if ( !v32 )
            {
              v44 = v24 - v21 + v42;
              if ( !v44 )
                v43 = v26 - v22 + (v21 - v42 < v24) + (v21 < v42);
            }
            LODWORD(v19) += 0x80000000;
            v42 = v32 != 0;
          }
          if ( (v43 & 0x800000) == 0 )
          {
            do
            {
              v45 = 2 * v43;
              v46 = v44;
              --v23;
              v44 *= 2;
              v43 = v45 | (v46 >> 31);
            }
            while ( (v45 & 0x800000) == 0 );
          }
          v22 = v43;
          v21 = v42 | v44;
        }
        else
        {
          v33 = v24 + v21;
          v22 = (__PAIR64__(v26, v24) + __PAIR64__(v22, v21)) >> 32;
          if ( (v22 & 0x1000000) != 0 )
          {
            v34 = v22;
            ++v23;
            v22 >>= 1;
            v21 = (v34 << 31) | ((v32 | (v33 << 31)) != 0) | (v33 >> 1);
          }
          else
          {
            v21 = (v32 != 0) | v33;
          }
        }
      }
      if ( (unsigned int)v23 <= 0xFD )
      {
        v35 = LODWORD(v19) | v22;
        if ( a3 )
        {
          if ( a3 != 1 )
          {
            if ( a3 == 3 )
            {
              v35 += v21 != 0 && LODWORD(v19) == 0;
            }
            else if ( a3 == 2 )
            {
              v35 += v21 != 0 && LODWORD(v19) != 0;
            }
          }
        }
        else
        {
          v36 = v21 >> 31;
          if ( v21 == 0x80000000 )
            v36 = v35 & 1;
          v35 += v36;
        }
        LODWORD(result) = v35 + (v23 << 23);
        return result;
      }
      if ( v23 > 125 )
      {
        if ( a3 )
        {
          switch ( a3 )
          {
            case 1:
              v22 = 2139095039;
              break;
            case 3:
              v22 = (LODWORD(v19) == 0) + 2139095039;
              break;
            case 2:
              v22 = 2139095039 - ((LODWORD(v19) == 0) - 1);
              break;
          }
        }
        else
        {
          v22 = 2139095040;
        }
        LODWORD(result) = v22 | LODWORD(v19);
        return result;
      }
      v38 = -v23;
      if ( !a1 )
      {
        if ( v23 >= -25 )
        {
          v41 = (v21 != 0) | (v22 << (v23 + 32));
          LODWORD(v40) = (v22 >> v38) + LODWORD(v19);
          if ( a3 )
            goto LABEL_75;
          if ( v41 == 0x80000000 )
          {
            v19 = v40;
            v50 = LOBYTE(v40) & 1;
LABEL_108:
            LODWORD(v40) = v50 + LODWORD(v19);
            return v40;
          }
          v19 = v40;
        }
        else
        {
          if ( a3 )
          {
            switch ( a3 )
            {
              case 1:
                v40 = v19;
                v39 = 0;
                if ( !v21 )
                  return v40;
                break;
              case 3:
                LODWORD(result) = (LODWORD(v19) == 0) | LODWORD(v19);
                return result;
              case 2:
                LODWORD(result) = (LODWORD(v19) != 0) | LODWORD(v19);
                return result;
              default:
                v39 = v22 << (v23 + 32);
                LODWORD(v40) = LODWORD(v19) + (v22 >> v38);
                if ( !v21 )
                  return v40;
                break;
            }
            v41 = v39 | 1;
LABEL_75:
            if ( a3 != 1 )
            {
              if ( a3 == 3 )
              {
                v57 = LODWORD(v19) == 0;
              }
              else
              {
                if ( a3 != 2 )
                  return v40;
                v57 = LODWORD(v19) != 0;
              }
              LODWORD(v40) += v57 && v41 != 0;
            }
            return v40;
          }
          v41 = v21 != 0;
        }
        v50 = v41 >> 31;
        goto LABEL_108;
      }
      if ( !a3 )
      {
        v47 = (v21 >> 31) + v22;
        if ( v47 > 0xFFFFFF )
        {
          v48 = v47 >> 1;
          if ( v23 == -1 )
          {
            v49 = 0;
LABEL_91:
            if ( v48 == 0x800000 && !v49 )
              LODWORD(v19) |= 0x800000u;
          }
        }
        return v19;
      }
      if ( a3 == 1 )
        return v19;
      v49 = v38 > 0x19;
      if ( a3 == 3 )
      {
        v55 = v21 != 0;
        v56 = LODWORD(v19) == 0;
      }
      else
      {
        if ( a3 != 2 )
        {
LABEL_104:
          v48 = v22 >> v38;
          goto LABEL_91;
        }
        v55 = v21 != 0;
        v56 = LODWORD(v19) != 0;
      }
      v22 += v55 && v56;
      goto LABEL_104;
    }
  }
  v51 = LODWORD(a6) & 0x7FFFFFFF;
  if ( a3 == 2 )
    v51 = (LODWORD(a6) ^ LODWORD(a4) ^ LODWORD(a5)) & 0x80000000;
  return *(float *)&v51;
}
