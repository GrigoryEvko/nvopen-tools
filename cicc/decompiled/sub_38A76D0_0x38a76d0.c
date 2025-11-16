// Function: sub_38A76D0
// Address: 0x38a76d0
//
__int64 __fastcall sub_38A76D0(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
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
  __int64 v19; // [rsp+10h] [rbp-90h] BYREF
  __int16 v20; // [rsp+18h] [rbp-88h]
  __int64 v21; // [rsp+20h] [rbp-80h] BYREF
  __int16 v22; // [rsp+28h] [rbp-78h]
  _QWORD v23[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v24; // [rsp+40h] [rbp-60h]
  _QWORD v25[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v26; // [rsp+60h] [rbp-40h]

  v6 = a1 + 8;
  v22 = 256;
  v19 = 0;
  v20 = 256;
  v21 = 0;
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
        if ( sub_2241AC0(a1 + 72, "name") )
        {
          if ( sub_2241AC0(a1 + 72, "type") )
          {
            v17 = *(_QWORD *)(a1 + 56);
            v23[1] = a1 + 72;
            v26 = 770;
            v23[0] = "invalid field '";
            v24 = 1027;
            v25[0] = v23;
            v25[1] = "'";
            v9 = sub_38814C0(v6, v17, (__int64)v25);
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
    v26 = 259;
    v25[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)v25) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v22 )
    {
      v12 = *(__int64 **)a1;
      if ( a3 )
        v13 = sub_15C24D0(v12, v19, v21, 1u, 1);
      else
        v13 = sub_15C24D0(v12, v19, v21, 0, 1);
      *a2 = v13;
    }
    else
    {
      v26 = 259;
      v25[0] = "missing required field 'type'";
      return (unsigned int)sub_38814C0(v6, v10, (__int64)v25);
    }
  }
  return v11;
}
