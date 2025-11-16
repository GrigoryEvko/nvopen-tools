// Function: sub_38A7000
// Address: 0x38a7000
//
__int64 __fastcall sub_38A7000(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
  int v8; // eax
  char v9; // al
  unsigned __int64 v10; // r14
  unsigned int v11; // r15d
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v19; // [rsp+10h] [rbp-D0h] BYREF
  __int16 v20; // [rsp+18h] [rbp-C8h]
  __int64 v21; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v22; // [rsp+28h] [rbp-B8h]
  _QWORD v23[4]; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD v24[4]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v25[2]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v26; // [rsp+80h] [rbp-60h]
  _QWORD v27[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v28; // [rsp+A0h] [rbp-40h]

  v6 = a1 + 8;
  v23[0] = 3;
  v23[1] = 0;
  v23[2] = 255;
  v24[0] = 0;
  v24[1] = 0;
  v24[2] = 0xFFFFFFFFLL;
  v19 = 0;
  v20 = 256;
  v21 = 0;
  v22 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v8 = *(_DWORD *)(a1 + 64);
  if ( v8 != 13 )
  {
    if ( v8 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "type") )
        {
          if ( sub_2241AC0(a1 + 72, "line") )
          {
            if ( sub_2241AC0(a1 + 72, "file") )
            {
              if ( sub_2241AC0(a1 + 72, "nodes") )
              {
                v17 = *(_QWORD *)(a1 + 56);
                v25[1] = a1 + 72;
                v25[0] = "invalid field '";
                v26 = 1027;
                v27[0] = v25;
                v27[1] = "'";
                v28 = 770;
                v9 = sub_38814C0(v6, v17, (__int64)v27);
              }
              else
              {
                v9 = sub_38A29E0(a1, (__int64)"nodes", 5, (__int64)&v21, a4, a5, a6);
              }
            }
            else
            {
              v9 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v19, a4, a5, a6);
            }
          }
          else
          {
            v9 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v24);
          }
        }
        else
        {
          v9 = sub_3889B40(a1, (__int64)"type", 4, (__int64)v23);
        }
        if ( v9 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v15 = sub_3887100(v6);
        *(_DWORD *)(a1 + 64) = v15;
      }
      while ( v15 == 372 );
    }
    v16 = *(_QWORD *)(a1 + 56);
    v28 = 259;
    v27[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)v27) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v20 )
    {
      v12 = *(__int64 **)a1;
      if ( a3 )
        v13 = sub_15C6E80(v12, v23[0], v24[0], v19, v21, 1u, 1);
      else
        v13 = sub_15C6E80(v12, v23[0], v24[0], v19, v21, 0, 1);
      *a2 = v13;
    }
    else
    {
      v28 = 259;
      v27[0] = "missing required field 'file'";
      return (unsigned int)sub_38814C0(v6, v10, (__int64)v27);
    }
  }
  return v11;
}
