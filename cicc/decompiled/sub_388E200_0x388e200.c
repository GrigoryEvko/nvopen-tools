// Function: sub_388E200
// Address: 0x388e200
//
__int64 __fastcall sub_388E200(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r15
  unsigned int v8; // r14d
  __int64 *v9; // rdi
  __int64 v10; // rax
  const char *v12; // rax
  int v13; // eax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rsi
  __int64 v17; // [rsp+10h] [rbp-D0h] BYREF
  __int16 v18; // [rsp+18h] [rbp-C8h]
  __int64 v19; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v20; // [rsp+28h] [rbp-B8h]
  __int64 v21; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v22; // [rsp+38h] [rbp-A8h]
  __int64 v23; // [rsp+40h] [rbp-A0h]
  _QWORD v24[4]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v25[2]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v26; // [rsp+80h] [rbp-60h]
  _QWORD v27[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v28; // [rsp+A0h] [rbp-40h]

  v3 = a1 + 8;
  v21 = 0;
  v22 = 0;
  v23 = 255;
  v24[0] = 0;
  v24[1] = 0;
  v24[2] = 0xFFFFFFFFLL;
  v17 = 0;
  v18 = 256;
  v19 = 0;
  v20 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v5 = *(_DWORD *)(a1 + 64);
  if ( v5 != 13 )
  {
    if ( v5 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "type") )
        {
          if ( sub_2241AC0(a1 + 72, "line") )
          {
            if ( sub_2241AC0(a1 + 72, "name") )
            {
              if ( sub_2241AC0(a1 + 72, "value") )
              {
                v15 = *(_QWORD *)(a1 + 56);
                v25[1] = a1 + 72;
                v25[0] = "invalid field '";
                v26 = 1027;
                v27[0] = v25;
                v27[1] = "'";
                v28 = 770;
                v6 = sub_38814C0(v3, v15, (__int64)v27);
              }
              else
              {
                v6 = sub_388B8F0(a1, (__int64)"value", 5, (__int64)&v19);
              }
            }
            else
            {
              v6 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v17);
            }
          }
          else
          {
            v6 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v24);
          }
        }
        else
        {
          v6 = sub_3889B40(a1, (__int64)"type", 4, (__int64)&v21);
        }
        if ( v6 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v13 = sub_3887100(v3);
        *(_DWORD *)(a1 + 64) = v13;
      }
      while ( v13 == 372 );
    }
    v14 = *(_QWORD *)(a1 + 56);
    v28 = 259;
    v27[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v3, v14, (__int64)v27) )
      return 1;
  }
LABEL_8:
  v7 = *(_QWORD *)(a1 + 56);
  v8 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v8 )
  {
    if ( (_BYTE)v22 )
    {
      if ( (_BYTE)v18 )
      {
        v9 = *(__int64 **)a1;
        if ( a3 )
          v10 = sub_15C68B0(v9, v21, v24[0], v17, v19, 1u, 1);
        else
          v10 = sub_15C68B0(v9, v21, v24[0], v17, v19, 0, 1);
        *a2 = v10;
        return v8;
      }
      HIBYTE(v28) = 1;
      v12 = "missing required field 'name'";
    }
    else
    {
      HIBYTE(v28) = 1;
      v12 = "missing required field 'type'";
    }
    v27[0] = v12;
    LOBYTE(v28) = 3;
    return (unsigned int)sub_38814C0(v3, v7, (__int64)v27);
  }
  return v8;
}
