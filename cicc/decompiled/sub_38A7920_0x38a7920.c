// Function: sub_38A7920
// Address: 0x38a7920
//
__int64 __fastcall sub_38A7920(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v7; // r13
  int v8; // eax
  char v9; // al
  unsigned __int64 v10; // r14
  unsigned int v11; // r15d
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v19; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v20; // [rsp+18h] [rbp-B8h]
  __int64 v21; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v22; // [rsp+28h] [rbp-A8h]
  __int64 v23; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v24; // [rsp+38h] [rbp-98h]
  _QWORD v25[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v26[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v27; // [rsp+70h] [rbp-60h]
  _QWORD v28[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v29; // [rsp+90h] [rbp-40h]

  v24 = 256;
  v7 = a1 + 8;
  v25[0] = 48;
  v25[1] = 0;
  v25[2] = 0xFFFF;
  v19 = 0;
  v20 = 256;
  v21 = 0;
  v22 = 256;
  v23 = 0;
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
        if ( sub_2241AC0(a1 + 72, "tag") )
        {
          if ( sub_2241AC0(a1 + 72, "name") )
          {
            if ( sub_2241AC0(a1 + 72, "type") )
            {
              if ( sub_2241AC0(a1 + 72, "value") )
              {
                v17 = *(_QWORD *)(a1 + 56);
                v26[1] = a1 + 72;
                v26[0] = "invalid field '";
                v27 = 1027;
                v28[0] = v26;
                v28[1] = "'";
                v29 = 770;
                v9 = sub_38814C0(v7, v17, (__int64)v28);
              }
              else
              {
                v9 = sub_38A29E0(a1, (__int64)"value", 5, (__int64)&v23, a4, a5, a6);
              }
            }
            else
            {
              v9 = sub_38A29E0(a1, (__int64)"type", 4, (__int64)&v21, a4, a5, a6);
            }
          }
          else
          {
            v9 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v19);
          }
        }
        else
        {
          v9 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)v25);
        }
        if ( v9 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v15 = sub_3887100(v7);
        *(_DWORD *)(a1 + 64) = v15;
      }
      while ( v15 == 372 );
    }
    v16 = *(_QWORD *)(a1 + 56);
    v29 = 259;
    v28[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v7, v16, (__int64)v28) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v24 )
    {
      v12 = *(__int64 **)a1;
      if ( a3 )
        v13 = sub_15C2A60(v12, v25[0], v19, v21, v23, 1u, 1);
      else
        v13 = sub_15C2A60(v12, v25[0], v19, v21, v23, 0, 1);
      *a2 = v13;
    }
    else
    {
      v29 = 259;
      v28[0] = "missing required field 'value'";
      return (unsigned int)sub_38814C0(v7, v10, (__int64)v28);
    }
  }
  return v11;
}
