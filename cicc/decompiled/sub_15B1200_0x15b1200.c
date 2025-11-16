// Function: sub_15B1200
// Address: 0x15b1200
//
__int64 __fastcall sub_15B1200(__int64 a1)
{
  unsigned __int64 *v1; // r14
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r14
  unsigned __int64 v4; // rax
  unsigned int v5; // r8d
  unsigned __int64 *v7; // r14
  unsigned __int64 *v8; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 *v9; // [rsp+8h] [rbp-38h] BYREF

  v1 = *(unsigned __int64 **)(a1 + 24);
  v2 = *(unsigned __int64 **)(a1 + 32);
  v8 = v1;
  if ( v1 == v2 )
    return 1;
  while ( 1 )
  {
    if ( v2 < &v1[(unsigned int)sub_15B11B0(&v8)] )
      return 0;
    v3 = v8;
    v4 = *v8;
    if ( *v8 == 159 )
    {
      if ( v2 != &v3[(unsigned int)sub_15B11B0(&v8)] )
      {
        v7 = v8;
        v9 = v8;
        v9 = &v7[(unsigned int)sub_15B11B0(&v9)];
        if ( *v9 != 4096 )
          return 0;
      }
      v3 = v8;
      goto LABEL_9;
    }
    if ( *v8 > 0x9F )
      break;
    if ( v4 <= 0x30 )
    {
      if ( v4 > 5 )
      {
        switch ( v4 )
        {
          case 6uLL:
          case 0x10uLL:
          case 0x12uLL:
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
          case 0x30uLL:
            goto LABEL_9;
          case 0x16uLL:
            if ( (unsigned int)((__int64)(*(_QWORD *)(a1 + 32) - *(_QWORD *)(a1 + 24)) >> 3) == 1 )
              return 0;
            goto LABEL_9;
          default:
            return 0;
        }
      }
      return 0;
    }
    if ( v4 - 147 > 2 )
      return 0;
LABEL_9:
    v1 = &v3[(unsigned int)sub_15B11B0(&v8)];
    v8 = v1;
    if ( v2 == v1 )
      return 1;
  }
  v5 = 0;
  if ( v4 == 4096 )
    LOBYTE(v5) = v2 == &v3[(unsigned int)sub_15B11B0(&v8)];
  return v5;
}
