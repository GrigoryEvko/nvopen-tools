// Function: sub_120BE50
// Address: 0x120be50
//
__int64 __fastcall sub_120BE50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  unsigned int v5; // r15d
  unsigned __int64 v6; // rsi
  int v9; // r14d
  int v10; // eax
  int v11; // eax
  int v12; // eax
  int v13; // edx
  unsigned int v14; // eax
  int v15; // [rsp+4h] [rbp-9Ch]
  _QWORD v16[2]; // [rsp+10h] [rbp-90h] BYREF
  char *v17; // [rsp+20h] [rbp-80h]
  __int64 v18; // [rsp+28h] [rbp-78h]
  __int16 v19; // [rsp+30h] [rbp-70h]
  _QWORD v20[2]; // [rsp+40h] [rbp-60h] BYREF
  char *v21; // [rsp+50h] [rbp-50h]
  __int16 v22; // [rsp+60h] [rbp-40h]

  v4 = a1 + 176;
  v5 = *(unsigned __int8 *)(a4 + 4);
  if ( !(_BYTE)v5 )
  {
    v9 = 0;
    v10 = sub_1205200(a1 + 176);
    for ( *(_DWORD *)(a1 + 240) = v10; ; *(_DWORD *)(a1 + 240) = v10 )
    {
      if ( v10 == 529 )
      {
        if ( !*(_BYTE *)(a1 + 332) )
        {
LABEL_12:
          v20[0] = "expected debug info flag";
          v22 = 259;
LABEL_13:
          v5 = 1;
          sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v20, 1);
          return v5;
        }
        LODWORD(v20[0]) = 0;
        v14 = sub_120BD00(a1, v20);
        v13 = v20[0];
        if ( (_BYTE)v14 )
          return v14;
        v12 = *(_DWORD *)(a1 + 240);
      }
      else
      {
        if ( v10 != 521 )
          goto LABEL_12;
        v11 = sub_AF1A40(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
        if ( !v11 )
        {
          v16[0] = "invalid debug info flag '";
          v17 = (char *)(a1 + 248);
          v19 = 1027;
          v20[0] = v16;
          v21 = "'";
          v22 = 770;
          goto LABEL_13;
        }
        v15 = v11;
        v12 = sub_1205200(v4);
        v13 = v15;
        *(_DWORD *)(a1 + 240) = v12;
      }
      v9 |= v13;
      if ( v12 != 15 )
      {
        *(_BYTE *)(a4 + 4) = 1;
        *(_DWORD *)a4 = v9;
        return v5;
      }
      v10 = sub_1205200(v4);
    }
  }
  v16[0] = "field '";
  v17 = "flags";
  v22 = 770;
  v6 = *(_QWORD *)(a1 + 232);
  v19 = 1283;
  v20[0] = v16;
  v18 = 5;
  v21 = "' cannot be specified more than once";
  sub_11FD800(a1 + 176, v6, (__int64)v20, 1);
  return v5;
}
