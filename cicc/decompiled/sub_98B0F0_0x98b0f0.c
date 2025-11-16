// Function: sub_98B0F0
// Address: 0x98b0f0
//
__int64 __fastcall sub_98B0F0(__int64 a1, _QWORD *a2, unsigned int a3)
{
  unsigned int v4; // r12d
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r15
  size_t v8; // rdx
  const void *v9; // r15
  size_t v10; // r14
  _BYTE *v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14[10]; // [rsp+0h] [rbp-50h] BYREF

  v4 = sub_98AE20(a1, v14, 8u, 0);
  if ( (_BYTE)v4 )
  {
    if ( v14[0] )
    {
      v5 = sub_AC52D0();
      v7 = v14[1];
      a2[1] = v6;
      if ( v7 > v6 )
      {
        v9 = (const void *)(v5 + v6);
        a2[1] = 0;
        *a2 = v5 + v6;
        if ( !(_BYTE)a3 )
          return v4;
      }
      else
      {
        v8 = v6 - v7;
        v9 = (const void *)(v5 + v7);
        *a2 = v9;
        v10 = v8;
        if ( v8 == -1 )
        {
          a2[1] = -1;
          if ( !(_BYTE)a3 )
            return v4;
          goto LABEL_12;
        }
        a2[1] = v8;
        if ( !(_BYTE)a3 )
          return v4;
        if ( v8 )
        {
LABEL_12:
          v12 = memchr(v9, 0, v8);
          if ( v12 )
          {
            v13 = v12 - (_BYTE *)v9;
            if ( v10 > v13 )
              v10 = v13;
          }
          goto LABEL_15;
        }
      }
      v10 = 0;
LABEL_15:
      *a2 = v9;
      a2[1] = v10;
      return v4;
    }
    if ( (_BYTE)a3 )
    {
      *a2 = 0;
      v4 = a3;
      a2[1] = 0;
    }
    else if ( v14[2] == 1 )
    {
      a2[1] = 1;
      *a2 = byte_3F871B3;
    }
    else
    {
      return 0;
    }
  }
  return v4;
}
