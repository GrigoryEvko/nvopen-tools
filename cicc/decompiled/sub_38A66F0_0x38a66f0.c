// Function: sub_38A66F0
// Address: 0x38a66f0
//
__int64 __fastcall sub_38A66F0(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
  int v8; // eax
  char v9; // al
  unsigned __int64 v10; // r15
  unsigned int v11; // r14d
  __int64 *v12; // rdi
  __int64 v13; // rax
  const char *v15; // rax
  int v16; // eax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  __int64 v20; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v21; // [rsp+18h] [rbp-A8h]
  __int64 v22; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v23; // [rsp+28h] [rbp-98h]
  __int64 v24; // [rsp+30h] [rbp-90h] BYREF
  __int64 v25; // [rsp+38h] [rbp-88h]
  __int64 v26; // [rsp+40h] [rbp-80h]
  _QWORD v27[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v28; // [rsp+60h] [rbp-60h]
  _QWORD v29[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v30; // [rsp+80h] [rbp-40h]

  v6 = a1 + 8;
  v23 = 256;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0xFFFFFFFFLL;
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
          if ( sub_2241AC0(a1 + 72, "file") )
          {
            if ( sub_2241AC0(a1 + 72, "discriminator") )
            {
              v18 = *(_QWORD *)(a1 + 56);
              v27[1] = a1 + 72;
              v30 = 770;
              v27[0] = "invalid field '";
              v28 = 1027;
              v29[0] = v27;
              v29[1] = "'";
              v9 = sub_38814C0(v6, v18, (__int64)v29);
            }
            else
            {
              v9 = sub_3889510(a1, (__int64)"discriminator", 13, (__int64)&v24);
            }
          }
          else
          {
            v9 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v22, a4, a5, a6);
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
        v16 = sub_3887100(v6);
        *(_DWORD *)(a1 + 64) = v16;
      }
      while ( v16 == 372 );
    }
    v17 = *(_QWORD *)(a1 + 56);
    v30 = 259;
    v29[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v17, (__int64)v29) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v21 )
    {
      if ( (_BYTE)v25 )
      {
        v12 = *(__int64 **)a1;
        if ( a3 )
          v13 = sub_15C0C90(v12, v20, v22, v24, 1u, 1);
        else
          v13 = sub_15C0C90(v12, v20, v22, v24, 0, 1);
        *a2 = v13;
        return v11;
      }
      HIBYTE(v30) = 1;
      v15 = "missing required field 'discriminator'";
    }
    else
    {
      HIBYTE(v30) = 1;
      v15 = "missing required field 'scope'";
    }
    v29[0] = v15;
    LOBYTE(v30) = 3;
    return (unsigned int)sub_38814C0(v6, v10, (__int64)v29);
  }
  return v11;
}
