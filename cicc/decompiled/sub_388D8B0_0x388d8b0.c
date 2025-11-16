// Function: sub_388D8B0
// Address: 0x388d8b0
//
__int64 __fastcall sub_388D8B0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v6; // eax
  char v7; // al
  unsigned int v8; // r13d
  __int64 *v9; // rdi
  __int64 v10; // rax
  int v12; // eax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rsi
  __int64 v15; // [rsp+10h] [rbp-100h] BYREF
  __int16 v16; // [rsp+18h] [rbp-F8h]
  _QWORD v17[4]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v18[4]; // [rsp+40h] [rbp-D0h] BYREF
  _QWORD v19[4]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v20[4]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD v21[2]; // [rsp+A0h] [rbp-70h] BYREF
  __int16 v22; // [rsp+B0h] [rbp-60h]
  _QWORD v23[2]; // [rsp+C0h] [rbp-50h] BYREF
  __int16 v24; // [rsp+D0h] [rbp-40h]

  v3 = a1 + 8;
  v16 = 256;
  v17[0] = 36;
  v17[1] = 0;
  v17[2] = 0xFFFF;
  v15 = 0;
  v18[0] = 0;
  v18[1] = 0;
  v18[2] = -1;
  v19[0] = 0;
  v19[1] = 0;
  v19[2] = 0xFFFFFFFFLL;
  v20[0] = 0;
  v20[1] = 0;
  v20[2] = 255;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v6 = *(_DWORD *)(a1 + 64);
  if ( v6 != 13 )
  {
    if ( v6 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "tag") )
        {
          if ( sub_2241AC0(a1 + 72, "name") )
          {
            if ( sub_2241AC0(a1 + 72, "size") )
            {
              if ( sub_2241AC0(a1 + 72, "align") )
              {
                if ( sub_2241AC0(a1 + 72, "encoding") )
                {
                  v14 = *(_QWORD *)(a1 + 56);
                  v21[0] = "invalid field '";
                  v21[1] = a1 + 72;
                  v23[0] = v21;
                  v22 = 1027;
                  v23[1] = "'";
                  v24 = 770;
                  v7 = sub_38814C0(v3, v14, (__int64)v23);
                }
                else
                {
                  v7 = sub_3889670(a1, (__int64)"encoding", 8, (__int64)v20);
                }
              }
              else
              {
                v7 = sub_3889510(a1, (__int64)"align", 5, (__int64)v19);
              }
            }
            else
            {
              v7 = sub_3889510(a1, (__int64)"size", 4, (__int64)v18);
            }
          }
          else
          {
            v7 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v15);
          }
        }
        else
        {
          v7 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)v17);
        }
        if ( v7 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v12 = sub_3887100(v3);
        *(_DWORD *)(a1 + 64) = v12;
      }
      while ( v12 == 372 );
    }
    v13 = *(_QWORD *)(a1 + 56);
    v24 = 259;
    v23[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v3, v13, (__int64)v23) )
      return 1;
  }
LABEL_8:
  v8 = sub_388AF10(a1, 13, "expected ')' here");
  if ( (_BYTE)v8 )
    return 1;
  v9 = *(__int64 **)a1;
  if ( a3 )
    v10 = sub_15BC830(v9, v17[0], v15, v18[0], v19[0], v20[0], 1u, 1);
  else
    v10 = sub_15BC830(v9, v17[0], v15, v18[0], v19[0], v20[0], 0, 1);
  *a2 = v10;
  return v8;
}
