// Function: sub_38A81E0
// Address: 0x38a81e0
//
__int64 __fastcall sub_38A81E0(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
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
  int v19; // [rsp+18h] [rbp-118h] BYREF
  char v20; // [rsp+1Ch] [rbp-114h]
  __int64 v21; // [rsp+20h] [rbp-110h] BYREF
  __int16 v22; // [rsp+28h] [rbp-108h]
  __int64 v23; // [rsp+30h] [rbp-100h] BYREF
  __int16 v24; // [rsp+38h] [rbp-F8h]
  __int64 v25; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v26; // [rsp+48h] [rbp-E8h]
  __int64 v27; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v28; // [rsp+58h] [rbp-D8h]
  _QWORD v29[4]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v30[4]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD v31[4]; // [rsp+A0h] [rbp-90h] BYREF
  _QWORD v32[2]; // [rsp+C0h] [rbp-70h] BYREF
  __int16 v33; // [rsp+D0h] [rbp-60h]
  _QWORD v34[2]; // [rsp+E0h] [rbp-50h] BYREF
  __int16 v35; // [rsp+F0h] [rbp-40h]

  v6 = a1 + 8;
  v24 = 256;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v29[0] = 0;
  v29[1] = 0;
  v29[2] = 0xFFFF;
  v25 = 0;
  v26 = 256;
  v30[0] = 0;
  v30[1] = 0;
  v30[2] = 0xFFFFFFFFLL;
  v27 = 0;
  v28 = 256;
  v19 = 0;
  v20 = 0;
  v31[0] = 0;
  v31[1] = 0;
  v31[2] = 0xFFFFFFFFLL;
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
            if ( sub_2241AC0(a1 + 72, "arg") )
            {
              if ( sub_2241AC0(a1 + 72, "file") )
              {
                if ( sub_2241AC0(a1 + 72, "line") )
                {
                  if ( sub_2241AC0(a1 + 72, "type") )
                  {
                    if ( sub_2241AC0(a1 + 72, "flags") )
                    {
                      if ( sub_2241AC0(a1 + 72, "align") )
                      {
                        v17 = *(_QWORD *)(a1 + 56);
                        v32[1] = a1 + 72;
                        v35 = 770;
                        v32[0] = "invalid field '";
                        v33 = 1027;
                        v34[0] = v32;
                        v34[1] = "'";
                        v9 = sub_38814C0(v6, v17, (__int64)v34);
                      }
                      else
                      {
                        v9 = sub_3889510(a1, (__int64)"align", 5, (__int64)v31);
                      }
                    }
                    else
                    {
                      v9 = sub_388BBA0(a1, (__int64)"flags", 5, (__int64)&v19);
                    }
                  }
                  else
                  {
                    v9 = sub_38A29E0(a1, (__int64)"type", 4, (__int64)&v27, a4, a5, a6);
                  }
                }
                else
                {
                  v9 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v30);
                }
              }
              else
              {
                v9 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v25, a4, a5, a6);
              }
            }
            else
            {
              v9 = sub_3889510(a1, (__int64)"arg", 3, (__int64)v29);
            }
          }
          else
          {
            v9 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v23);
          }
        }
        else
        {
          v9 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v21, a4, a5, a6);
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
    v35 = 259;
    v34[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)v34) )
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
        v13 = sub_15C37C0(v12, v21, v23, v25, v30[0], v27, v29[0], v19, v31[0], 1u, 1);
      else
        v13 = sub_15C37C0(v12, v21, v23, v25, v30[0], v27, v29[0], v19, v31[0], 0, 1);
      *a2 = v13;
    }
    else
    {
      v35 = 259;
      v34[0] = "missing required field 'scope'";
      return (unsigned int)sub_38814C0(v6, v10, (__int64)v34);
    }
  }
  return v11;
}
