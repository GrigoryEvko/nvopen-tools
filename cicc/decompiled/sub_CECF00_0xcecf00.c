// Function: sub_CECF00
// Address: 0xcecf00
//
float __fastcall sub_CECF00(int a1, int a2, int a3, float a4, float a5)
{
  unsigned int v7; // eax
  unsigned int v8; // edx
  int v9; // r10d
  unsigned int v10; // esi
  int v11; // r11d
  unsigned int v12; // ebx
  int v13; // ecx
  int v14; // r12d
  int v15; // ecx
  int v16; // r9d
  __int64 v17; // rax
  unsigned int v18; // esi
  unsigned int v19; // r11d
  unsigned int v20; // edx
  unsigned int v21; // ecx
  bool v22; // cc
  float result; // xmm0_4
  unsigned int v24; // ecx
  int v25; // edx
  bool v26; // sf
  unsigned int v27; // r9d
  unsigned int v28; // eax
  unsigned int v29; // ecx
  char v30; // al
  unsigned int v31; // edi
  unsigned int v32; // edx
  int v33; // edx
  int v34; // edx
  unsigned int v35; // edx
  int v36; // eax
  unsigned int v37; // eax
  unsigned int v38; // ecx
  unsigned int v39; // esi
  int v40; // eax
  int v41; // eax

  v7 = LODWORD(a4);
  v8 = LODWORD(a5);
  if ( a1 )
  {
    if ( 2 * LODWORD(a4) < (unsigned int)&loc_1000000 )
      v7 = LODWORD(a4) & 0x80000000;
    if ( 2 * LODWORD(a5) < (unsigned int)&loc_1000000 )
      v8 = LODWORD(a5) & 0x80000000;
  }
  v9 = (unsigned __int8)(v7 >> 23);
  v10 = (unsigned __int8)(v8 >> 23);
  v11 = v9 - 1;
  v12 = v10 - 1;
  v13 = v9 - 1;
  v14 = v10 - 1;
  if ( (unsigned int)(v9 - 1) <= 0xFD && v12 <= 0xFD )
    goto LABEL_8;
  if ( (v7 & 0x7FFFFFFF) == 0 )
  {
    if ( v10 == 255 )
    {
      if ( 2 * v8 == -16777216 )
      {
        v33 = a2 == 0 ? -4194304 : 0x7FFFFFFF;
      }
      else
      {
        v33 = v8 | 0x400000;
        if ( a2 )
          v33 = 0x7FFFFFFF;
      }
      return *(float *)&v33;
    }
    else
    {
      LODWORD(result) = (v8 ^ v7) & 0x80000000;
    }
    return result;
  }
  if ( (v8 & 0x7FFFFFFF) != 0 )
  {
    if ( v10 != 255 && v9 != 255 )
    {
      if ( !(unsigned __int8)(v7 >> 23) )
      {
        v39 = v7 & 0x80000000 | v10;
        v40 = v7 << 8;
        if ( v40 >= 0 )
        {
          while ( 1 )
          {
            v9 = v11;
            v26 = (v40 & 0x40000000) != 0;
            v40 *= 2;
            if ( v26 )
              break;
            --v11;
          }
        }
        v14 = (v39 & 0x7FFFFFFF) - 1;
        v7 = v39 & 0x80000000 | ((unsigned int)v40 >> 8);
        v13 = v9;
        goto LABEL_8;
      }
      if ( !(unsigned __int8)(v8 >> 23) )
      {
        v24 = v9 | v8 & 0x80000000;
        v25 = v8 << 8;
        if ( v25 >= 0 )
        {
          while ( 1 )
          {
            v10 = v12;
            v26 = (v25 & 0x40000000) != 0;
            v25 *= 2;
            if ( v26 )
              break;
            --v12;
          }
        }
        v14 = v10;
        v27 = v24 & 0x80000000;
        v13 = (v24 & 0x7FFFFFFF) - 1;
        v8 = v27 | ((unsigned int)v25 >> 8);
LABEL_8:
        v15 = v14 + v13;
        v16 = v8 ^ v7;
        v17 = ((v8 << 8) | 0x80000000) * (v7 & 0xFFFFFF | 0x800000LL);
        v18 = v16 & 0x80000000;
        v19 = v17;
        v20 = HIDWORD(v17);
        if ( HIDWORD(v17) <= 0x7FFFFF )
        {
          v19 = 2 * v17;
          v21 = v15 - 126;
          v20 = v17 >> 31;
          v22 = (int)v21 <= 253;
          if ( v21 <= 0xFD )
          {
LABEL_10:
            v18 = (v21 << 23) + (v20 | v18);
            if ( a3 )
            {
              if ( a3 != 1 )
              {
                if ( a3 == 3 )
                {
                  v18 += v19 != 0 && v16 >= 0;
                }
                else if ( a3 == 2 )
                {
                  v18 += (v19 != 0) & ((unsigned int)v16 >> 31);
                }
              }
            }
            else
            {
              v28 = v19 >> 31;
              if ( v19 == 0x80000000 )
                v28 = v18 & 1;
              v18 += v28;
            }
            return *(float *)&v18;
          }
        }
        else
        {
          v21 = v15 - 125;
          v22 = (int)v21 <= 253;
          if ( v21 <= 0xFD )
            goto LABEL_10;
        }
        if ( v22 )
        {
          v29 = -v21;
          v30 = 25;
          if ( v29 <= 0x19 )
            v30 = v29;
          if ( a1 )
          {
            if ( a3 )
            {
              switch ( a3 )
              {
                case 1:
                  return *(float *)&v18;
                case 3:
                  v35 = (v19 != 0 && v16 >= 0) + v20;
                  break;
                case 2:
                  v35 = ((v19 != 0) & ((unsigned int)v16 >> 31)) + v20;
                  break;
                default:
                  return *(float *)&v18;
              }
            }
            else
            {
              v38 = v19 >> 31;
              if ( v19 == 0x80000000 )
                v38 = v20 & 1;
              v35 = v38 + v20;
            }
            if ( v35 >> v30 == 0x800000 )
              v18 |= 0x800000u;
            return *(float *)&v18;
          }
          v31 = (v19 != 0) | (v20 << (32 - v30));
          v32 = v18 + (v20 >> v30);
          switch ( a3 )
          {
            case 0:
              v37 = v31 >> 31;
              if ( v31 == 0x80000000 )
                v37 = v32 & 1;
              v18 |= v37 + v32;
              return *(float *)&v18;
            case 1:
              goto LABEL_46;
            case 3:
              v18 |= (v31 != 0 && v16 >= 0) + v32;
              return *(float *)&v18;
            case 2:
              v18 |= ((v31 != 0) & ((unsigned int)v16 >> 31)) + v32;
              break;
            default:
LABEL_46:
              v18 |= v32;
              break;
          }
          return *(float *)&v18;
        }
        if ( a3 )
        {
          v18 |= 0x7F7FFFFFu;
          if ( a3 != 1 )
          {
            if ( a3 == 3 )
            {
              v18 = ((v16 >> 31) & 0x7FFFFFFF) + 2139095040;
            }
            else
            {
              v18 = v20;
              if ( a3 == 2 )
                v18 = ((v16 >> 31) & 0x80000001) + 2139095039;
            }
          }
          return *(float *)&v18;
        }
LABEL_59:
        v18 |= 0x7F800000u;
        return *(float *)&v18;
      }
    }
    if ( 2 * v7 > 0xFF000000 )
    {
      v36 = v7 | 0x400000;
      if ( a2 )
        v36 = 0x7FFFFFFF;
      return *(float *)&v36;
    }
    else
    {
      if ( 2 * v8 <= 0xFF000000 )
      {
        v18 = (v8 ^ v7) & 0x80000000;
        goto LABEL_59;
      }
      v34 = v8 | 0x400000;
      if ( a2 )
        v34 = 0x7FFFFFFF;
      return *(float *)&v34;
    }
  }
  else if ( v9 == 255 )
  {
    if ( 2 * v7 == -16777216 )
    {
      v41 = a2 == 0 ? -4194304 : 0x7FFFFFFF;
    }
    else
    {
      v41 = v7 | 0x400000;
      if ( a2 )
        v41 = 0x7FFFFFFF;
    }
    return *(float *)&v41;
  }
  else
  {
    LODWORD(result) = (v8 ^ v7) & 0x80000000;
  }
  return result;
}
