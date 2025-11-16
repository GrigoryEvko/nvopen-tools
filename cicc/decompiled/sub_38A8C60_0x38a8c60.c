// Function: sub_38A8C60
// Address: 0x38a8c60
//
__int64 __fastcall sub_38A8C60(__int64 a1, unsigned int **a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
  int v9; // eax
  char v10; // al
  unsigned int v11; // r13d
  __int64 *v12; // rdi
  unsigned int *v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v18; // [rsp+10h] [rbp-100h] BYREF
  __int16 v19; // [rsp+18h] [rbp-F8h]
  __int64 v20; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v21; // [rsp+28h] [rbp-E8h]
  __int64 v22; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v23; // [rsp+38h] [rbp-D8h]
  __int64 v24; // [rsp+40h] [rbp-D0h] BYREF
  __int16 v25; // [rsp+48h] [rbp-C8h]
  __int64 v26; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v27; // [rsp+58h] [rbp-B8h]
  _QWORD v28[4]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v29[4]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD v30[2]; // [rsp+A0h] [rbp-70h] BYREF
  __int16 v31; // [rsp+B0h] [rbp-60h]
  _QWORD v32[2]; // [rsp+C0h] [rbp-50h] BYREF
  __int16 v33; // [rsp+D0h] [rbp-40h]

  v6 = a1 + 8;
  v21 = 256;
  v18 = 0;
  v19 = 256;
  v20 = 0;
  v28[0] = 0;
  v28[1] = 0;
  v28[2] = 0xFFFFFFFFLL;
  v22 = 0;
  v23 = 256;
  v24 = 0;
  v25 = 256;
  v29[0] = 0;
  v29[1] = 0;
  v29[2] = 0xFFFFFFFFLL;
  v26 = 0;
  v27 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v9 = *(_DWORD *)(a1 + 64);
  if ( v9 != 13 )
  {
    if ( v9 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "name") )
        {
          if ( sub_2241AC0(a1 + 72, "file") )
          {
            if ( sub_2241AC0(a1 + 72, "line") )
            {
              if ( sub_2241AC0(a1 + 72, "setter") )
              {
                if ( sub_2241AC0(a1 + 72, "getter") )
                {
                  if ( sub_2241AC0(a1 + 72, "attributes") )
                  {
                    if ( sub_2241AC0(a1 + 72, "type") )
                    {
                      v17 = *(_QWORD *)(a1 + 56);
                      v30[1] = a1 + 72;
                      v33 = 770;
                      v30[0] = "invalid field '";
                      v31 = 1027;
                      v32[0] = v30;
                      v32[1] = "'";
                      v10 = sub_38814C0(v6, v17, (__int64)v32);
                    }
                    else
                    {
                      v10 = sub_38A29E0(a1, (__int64)"type", 4, (__int64)&v26, a4, a5, a6);
                    }
                  }
                  else
                  {
                    v10 = sub_3889510(a1, (__int64)"attributes", 10, (__int64)v29);
                  }
                }
                else
                {
                  v10 = sub_388B8F0(a1, (__int64)"getter", 6, (__int64)&v24);
                }
              }
              else
              {
                v10 = sub_388B8F0(a1, (__int64)"setter", 6, (__int64)&v22);
              }
            }
            else
            {
              v10 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v28);
            }
          }
          else
          {
            v10 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v20, a4, a5, a6);
          }
        }
        else
        {
          v10 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v18);
        }
        if ( v10 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v15 = sub_3887100(v6);
        *(_DWORD *)(a1 + 64) = v15;
      }
      while ( v15 == 372 );
    }
    v16 = *(_QWORD *)(a1 + 56);
    v33 = 259;
    v32[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)v32) )
      return 1;
  }
LABEL_8:
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( (_BYTE)v11 )
    return 1;
  v12 = *(__int64 **)a1;
  if ( a3 )
    v13 = sub_15C5B60(v12, v18, v20, v28[0], v22, v24, v29[0], v26, 1u, 1);
  else
    v13 = sub_15C5B60(v12, v18, v20, v28[0], v22, v24, v29[0], v26, 0, 1);
  *a2 = v13;
  return v11;
}
