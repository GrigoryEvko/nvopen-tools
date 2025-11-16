// Function: sub_1C3F110
// Address: 0x1c3f110
//
float __fastcall sub_1C3F110(int a1, int a2, int a3, float a4, float a5)
{
  float v8; // edx
  unsigned int v9; // eax
  unsigned int v10; // ecx
  unsigned int v11; // esi
  float v12; // ecx
  int v13; // edi
  int v14; // esi
  unsigned int v15; // r10d
  unsigned int v16; // ecx
  int v17; // edi
  unsigned int v18; // esi
  int v19; // edx
  int v20; // esi
  unsigned int v21; // eax
  char v22; // dl
  char v23; // cl
  float v24; // esi
  unsigned int v26; // esi
  unsigned int v27; // ecx
  int v28; // eax
  unsigned int v29; // ecx
  int v30; // edx
  unsigned int v31; // esi
  float v32; // esi
  float v33; // esi
  int v34; // edx
  float v35; // eax
  float v36; // edx
  int v37; // eax
  bool v38; // zf
  int v39; // eax
  unsigned int v40; // esi
  int v41; // edx
  unsigned int v42; // edi
  bool v43; // sf
  unsigned int v44; // ebx
  int v45; // eax
  unsigned int v46; // r11d
  int v47; // eax
  unsigned int v48; // edx

  v8 = a4;
  *(float *)&v9 = a5;
  v10 = 2 * LODWORD(a4);
  if ( a1 )
  {
    if ( v10 <= 0xFFFFFF )
    {
      LODWORD(v8) = LODWORD(a4) & 0x80000000;
      v10 = 0;
    }
    v11 = 2 * LODWORD(a5);
    if ( (unsigned int)(2 * LODWORD(a5)) <= 0xFFFFFF )
    {
      v9 = LODWORD(a5) & 0x80000000;
      goto LABEL_5;
    }
  }
  else
  {
    v11 = 2 * LODWORD(a5);
  }
  if ( v10 < v11 )
  {
    v12 = v8;
    v8 = a5;
    *(float *)&v9 = v12;
  }
LABEL_5:
  v13 = (unsigned __int8)(LODWORD(v8) >> 23);
  v14 = (unsigned __int8)(v9 >> 23);
  v15 = v13 - 1;
  v16 = v14 - 1;
  if ( (unsigned int)(v13 - 1) <= 0xFD && v16 <= 0xFD )
  {
    v17 = v13 - v14;
    v18 = LODWORD(v8) & 0x80FFFFFF;
    goto LABEL_8;
  }
  v26 = 2 * v9;
  if ( !(2 * v9) )
  {
    if ( a3 == 2 )
    {
      v27 = 2 * LODWORD(v8);
      if ( !(2 * LODWORD(v8)) )
      {
        LODWORD(v8) |= v9;
        v27 = 2 * LODWORD(v8);
      }
    }
    else
    {
      if ( LODWORD(v8) == 0x80000000 )
        return *(float *)&v9;
      v27 = 2 * LODWORD(v8);
    }
    if ( v27 > 0xFF000000 )
    {
      LODWORD(v8) |= 0x400000u;
      if ( a2 )
        return NAN;
    }
    return v8;
  }
  if ( v16 == 254 || v13 == 255 )
  {
    if ( (unsigned int)(2 * LODWORD(v8)) <= 0xFF000000 )
    {
      if ( v26 > 0xFF000000 )
      {
        v37 = v9 | 0x400000;
        if ( a2 )
          *(float *)&v37 = NAN;
        return *(float *)&v37;
      }
      else if ( 2 * LODWORD(v8) == -16777216 && v26 == -16777216 )
      {
        if ( LODWORD(v8) != v9 )
          v9 = a2 == 0 ? -4194304 : 0x7FFFFFFF;
        return *(float *)&v9;
      }
      else
      {
        if ( v26 == -16777216 )
          return *(float *)&v9;
        return v8;
      }
    }
    else
    {
      v34 = LODWORD(v8) | 0x400000;
      if ( a2 )
        *(float *)&v34 = NAN;
      return *(float *)&v34;
    }
  }
  if ( (unsigned __int8)(LODWORD(v8) >> 23) )
  {
    v18 = LODWORD(v8) & 0x80FFFFFF;
  }
  else
  {
    v40 = LODWORD(v8) & 0x80000000;
    v41 = LODWORD(v8) << 8;
    if ( v41 < 0 )
    {
      v42 = 0;
    }
    else
    {
      do
      {
        v42 = v15--;
        v43 = (v41 & 0x40000000) != 0;
        v41 *= 2;
      }
      while ( !v43 );
    }
    v15 = v42;
    LODWORD(v8) = v40 | ((unsigned int)v41 >> 8);
    v18 = LODWORD(v8);
  }
  if ( v16 == -1 )
  {
    v44 = v9 & 0x80000000;
    v45 = v9 << 8;
    if ( v45 < 0 )
    {
      v17 = v15;
      LOBYTE(v16) = 0;
    }
    else
    {
      do
      {
        v46 = v16--;
        v43 = (v45 & 0x40000000) != 0;
        v45 *= 2;
      }
      while ( !v43 );
      LOBYTE(v16) = v46;
      v17 = v15 - v46;
    }
    v9 = v44 | ((unsigned int)v45 >> 8);
  }
  else
  {
    v17 = v15 - v16;
  }
LABEL_8:
  v19 = v9 ^ LODWORD(v8);
  v20 = v18 | 0x800000;
  v21 = v9 & 0x7FFFFF | 0x800000;
  if ( (unsigned int)v17 > 0x19 )
  {
    if ( v19 >= 0 )
    {
      v22 = 31;
      v23 = 1;
      goto LABEL_11;
    }
    v28 = -2 * v21;
LABEL_42:
    if ( (--v20 & 0x800000) == 0 )
    {
LABEL_43:
      v29 = v20 & 0x80000000;
      do
      {
        v30 = 2 * v20;
        v31 = v28;
        --v15;
        v28 *= 2;
        v20 = v30 | (v31 >> 31);
      }
      while ( (v30 & 0x800000) == 0 );
      v20 |= v29;
      goto LABEL_46;
    }
    goto LABEL_54;
  }
  v23 = v16 + 32 - v15;
  if ( v19 < 0 )
  {
    if ( v17 )
    {
      v48 = v21 << v23;
      v20 -= v21 >> v17;
      v28 = -(v21 << v23);
      if ( v48 )
        goto LABEL_42;
    }
    else
    {
      v20 -= v21;
    }
    v28 = v20 & 0x800000;
    if ( (v20 & 0x800000) == 0 )
    {
      LODWORD(v36) = 2 * v20;
      if ( !(2 * v20) )
      {
        if ( a3 == 2 )
          return -0.0;
        return v36;
      }
      goto LABEL_43;
    }
    v28 = 0;
LABEL_54:
    if ( v15 <= 0xFD )
    {
      LODWORD(v33) = (v15 << 23) + v20;
      if ( a3 )
      {
        if ( a3 != 1 )
        {
          if ( a3 == 3 )
          {
            LODWORD(v33) += v28 != 0 && v33 >= 0.0;
          }
          else if ( a3 == 2 )
          {
            LODWORD(v33) += (v28 != 0) & (LODWORD(v33) >> 31);
          }
        }
      }
      else
      {
        LODWORD(v33) += (unsigned int)v28 >> 31;
      }
      return v33;
    }
    goto LABEL_67;
  }
  v22 = 0;
  if ( !v17 )
    goto LABEL_12;
  v22 = v17;
LABEL_11:
  v17 = v21 << v23;
LABEL_12:
  v20 += v21 >> v22;
  if ( (v20 & 0x1000000) == 0 )
  {
    if ( v15 <= 0xFD )
    {
      LODWORD(v24) = (v15 << 23) + v20;
      if ( a3 )
      {
        if ( a3 != 1 )
        {
          if ( a3 == 3 )
          {
            LODWORD(v24) += v17 != 0 && v24 >= 0.0;
          }
          else if ( a3 == 2 )
          {
            LODWORD(v24) += (v17 != 0) & (LODWORD(v24) >> 31);
          }
        }
        return v24;
      }
      if ( v17 < 0 )
      {
        v47 = LOBYTE(v24) & 1;
        if ( v17 != 0x80000000 )
          v47 = 1;
        LODWORD(v24) += v47;
        return v24;
      }
      return v24;
    }
LABEL_67:
    LODWORD(v35) = v20 & 0x80000000;
    if ( (int)v15 <= 253 )
    {
      if ( !a1 )
        LODWORD(v35) |= (v20 & 0xFFFFFFu) >> -(char)v15;
      return v35;
    }
    else
    {
      if ( a3 )
      {
        switch ( a3 )
        {
          case 1:
            v20 = LODWORD(v35) | 0x7F7FFFFF;
            break;
          case 2:
            v20 = ((v20 >> 31) & 0x80000001) + 2139095039;
            break;
          case 3:
            v20 = ((v20 >> 31) & 0x7FFFFFFF) + 2139095040;
            break;
        }
      }
      else
      {
        v20 = LODWORD(v35) | 0x7F800000;
      }
      return *(float *)&v20;
    }
  }
  ++v15;
  v28 = ((unsigned int)v17 >> 1) | (v20 << 31);
  v20 = v20 & 0x80000000 | ((unsigned int)v20 >> 1) & 0xBFFFFFFF;
LABEL_46:
  if ( v15 > 0xFD )
    goto LABEL_67;
  LODWORD(v32) = (v15 << 23) + v20;
  if ( a3 )
  {
    if ( a3 != 1 )
    {
      if ( a3 == 3 )
      {
        LODWORD(v32) += v28 != 0 && v32 >= 0.0;
      }
      else if ( a3 == 2 )
      {
        LODWORD(v32) += (v28 != 0) & (LODWORD(v32) >> 31);
      }
    }
  }
  else
  {
    if ( v28 >= 0 )
      return v32;
    v38 = v28 == 0x80000000;
    v39 = 1;
    if ( v38 )
      v39 = LOBYTE(v32) & 1;
    LODWORD(v32) += v39;
  }
  return v32;
}
