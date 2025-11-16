// Function: sub_38A6D50
// Address: 0x38a6d50
//
__int64 __fastcall sub_38A6D50(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
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
  __int16 v19; // [rsp+1Eh] [rbp-92h] BYREF
  __int64 v20; // [rsp+20h] [rbp-90h] BYREF
  __int16 v21; // [rsp+28h] [rbp-88h]
  __int64 v22; // [rsp+30h] [rbp-80h] BYREF
  __int16 v23; // [rsp+38h] [rbp-78h]
  _QWORD v24[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v25; // [rsp+50h] [rbp-60h]
  _QWORD v26[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v27; // [rsp+70h] [rbp-40h]

  v6 = a1 + 8;
  v23 = 256;
  v20 = 0;
  v21 = 256;
  v22 = 0;
  v19 = 0;
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
        if ( sub_2241AC0(a1 + 72, "scope") )
        {
          if ( sub_2241AC0(a1 + 72, "name") )
          {
            if ( sub_2241AC0(a1 + 72, "exportSymbols") )
            {
              v17 = *(_QWORD *)(a1 + 56);
              v24[1] = a1 + 72;
              v27 = 770;
              v24[0] = "invalid field '";
              v25 = 1027;
              v26[0] = v24;
              v26[1] = "'";
              v9 = sub_38814C0(v6, v17, (__int64)v26);
            }
            else
            {
              v9 = sub_3887760(a1, (__int64)"exportSymbols", 13, (__int64)&v19);
            }
          }
          else
          {
            v9 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v22);
          }
        }
        else
        {
          v9 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v20, a4, a5, a6);
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
    v27 = 259;
    v26[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)v26) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v21 )
    {
      v12 = *(__int64 **)a1;
      if ( a3 )
        v13 = sub_15C1230(v12, v20, v22, v19, 1u, 1);
      else
        v13 = sub_15C1230(v12, v20, v22, v19, 0, 1);
      *a2 = v13;
    }
    else
    {
      v27 = 259;
      v26[0] = "missing required field 'scope'";
      return (unsigned int)sub_38814C0(v6, v10, (__int64)v26);
    }
  }
  return v11;
}
