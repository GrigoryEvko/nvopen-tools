// Function: sub_AF4230
// Address: 0xaf4230
//
bool __fastcall sub_AF4230(__int64 a1)
{
  unsigned __int64 *v1; // r14
  unsigned __int64 *v2; // r12
  unsigned __int64 *v3; // r14
  unsigned __int64 v4; // rax
  bool result; // al
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r14
  unsigned int v8; // eax
  unsigned __int64 *v9; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 *v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = *(unsigned __int64 **)(a1 + 16);
  v2 = *(unsigned __int64 **)(a1 + 24);
  v9 = v1;
  if ( v1 == v2 )
    return 1;
  while ( 1 )
  {
    if ( v2 < &v1[(unsigned int)sub_AF4160(&v9)] )
      return 0;
    v3 = v9;
    v4 = *v9;
    if ( *v9 - 80 <= 0x3F )
      return 1;
    if ( v4 == 159 )
    {
      if ( v2 != &v3[(unsigned int)sub_AF4160(&v9)] )
      {
        v7 = v9;
        v10[0] = v9;
        v10[0] = &v7[(unsigned int)sub_AF4160(v10)];
        if ( *v10[0] != 4096 )
          return 0;
      }
      v3 = v9;
      goto LABEL_12;
    }
    if ( v4 <= 0x9F )
    {
      if ( v4 <= 0x30 )
      {
        if ( v4 > 5 )
        {
          switch ( v4 )
          {
            case 6uLL:
            case 0x10uLL:
            case 0x11uLL:
            case 0x12uLL:
            case 0x14uLL:
            case 0x18uLL:
            case 0x1AuLL:
            case 0x1BuLL:
            case 0x1CuLL:
            case 0x1DuLL:
            case 0x1EuLL:
            case 0x20uLL:
            case 0x21uLL:
            case 0x22uLL:
            case 0x23uLL:
            case 0x24uLL:
            case 0x25uLL:
            case 0x26uLL:
            case 0x27uLL:
            case 0x29uLL:
            case 0x2AuLL:
            case 0x2BuLL:
            case 0x2CuLL:
            case 0x2DuLL:
            case 0x2EuLL:
            case 0x30uLL:
              goto LABEL_12;
            case 0x16uLL:
              if ( (unsigned int)((__int64)(*(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 16)) >> 3) == 1 )
                return 0;
              goto LABEL_12;
            default:
              return 0;
          }
        }
        return 0;
      }
      if ( v4 > 0x95 )
      {
        if ( v4 != 151 )
          return 0;
      }
      else if ( v4 <= 0x91 && v4 != 144 )
      {
        return 0;
      }
      goto LABEL_12;
    }
    if ( v4 == 4099 )
      break;
    if ( v4 <= 0x1003 )
    {
      if ( v4 == 4096 )
        return v2 == &v3[(unsigned int)sub_AF4160(&v9)];
      if ( v4 - 4097 > 1 )
        return 0;
    }
    else if ( v4 - 4100 > 3 )
    {
      return 0;
    }
LABEL_12:
    v1 = &v3[(unsigned int)sub_AF4160(&v9)];
    v9 = v1;
    if ( v2 == v1 )
      return 1;
  }
  v6 = *(unsigned __int64 **)(a1 + 16);
  v10[0] = v6;
  if ( *v6 == 4101 && !v6[1] )
  {
    v8 = sub_AF4160(v10);
    v3 = v9;
    v6 += v8;
  }
  result = 0;
  if ( v6 == v3 )
    return v6[1] == 1;
  return result;
}
