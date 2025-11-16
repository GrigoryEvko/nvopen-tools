// Function: sub_10E40A0
// Address: 0x10e40a0
//
__int64 __fastcall sub_10E40A0(_QWORD **a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // edx
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // rax
  __int64 v10; // r14
  __int64 v11; // rdx
  _BYTE *v12; // rax
  unsigned int v13; // r14d
  char v14; // r15
  unsigned int v15; // r14d
  __int64 v16; // rax
  unsigned int v17; // r15d
  int v18; // [rsp+Ch] [rbp-34h]

  v2 = *a2;
  if ( (unsigned __int8)v2 <= 0x1Cu )
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v4 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFF7) != 0x11 && (v4 & 0xFFFD) != 0xD )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)v2 > 0x36u )
      return 0;
    v3 = 0x40540000000000LL;
    v4 = (unsigned __int8)v2 - 29;
    if ( !_bittest64(&v3, v2) )
      return 0;
  }
  if ( v4 == 15 && (a2[1] & 4) != 0 )
  {
    v5 = *((_QWORD *)a2 - 8);
    if ( *(_BYTE *)v5 == 17 )
    {
      v6 = *(_DWORD *)(v5 + 32);
      if ( v6 <= 0x40 )
        v7 = *(_QWORD *)(v5 + 24) == 0;
      else
        v7 = v6 == (unsigned int)sub_C444A0(v5 + 24);
    }
    else
    {
      v10 = *(_QWORD *)(v5 + 8);
      v11 = (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17;
      if ( (unsigned int)v11 > 1 || *(_BYTE *)v5 > 0x15u )
        return 0;
      v12 = sub_AD7630(*((_QWORD *)a2 - 8), 0, v11);
      if ( !v12 || *v12 != 17 )
      {
        if ( *(_BYTE *)(v10 + 8) == 17 )
        {
          v18 = *(_DWORD *)(v10 + 32);
          if ( v18 )
          {
            v14 = 0;
            v15 = 0;
            while ( 1 )
            {
              v16 = sub_AD69F0((unsigned __int8 *)v5, v15);
              if ( !v16 )
                break;
              if ( *(_BYTE *)v16 != 13 )
              {
                if ( *(_BYTE *)v16 != 17 )
                  return 0;
                v17 = *(_DWORD *)(v16 + 32);
                if ( v17 <= 0x40 )
                {
                  if ( *(_QWORD *)(v16 + 24) )
                    return 0;
                }
                else if ( v17 != (unsigned int)sub_C444A0(v16 + 24) )
                {
                  return 0;
                }
                v14 = 1;
              }
              if ( v18 == ++v15 )
              {
                if ( v14 )
                  goto LABEL_10;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v13 = *((_DWORD *)v12 + 8);
      if ( v13 <= 0x40 )
      {
        if ( *((_QWORD *)v12 + 3) )
          return 0;
LABEL_10:
        if ( *a1 )
          **a1 = v5;
        v8 = *((_QWORD *)a2 - 4);
        if ( v8 )
        {
          *a1[1] = v8;
          return 1;
        }
        return 0;
      }
      v7 = v13 == (unsigned int)sub_C444A0((__int64)(v12 + 24));
    }
    if ( !v7 )
      return 0;
    goto LABEL_10;
  }
  return 0;
}
