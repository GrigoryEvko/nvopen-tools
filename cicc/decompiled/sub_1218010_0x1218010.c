// Function: sub_1218010
// Address: 0x1218010
//
__int64 __fastcall sub_1218010(__int64 a1, __int64 **a2, __int64 a3, unsigned int a4, unsigned __int64 *a5)
{
  unsigned int v7; // eax
  unsigned int v8; // r13d
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r8
  unsigned int v11; // r14d
  int v13; // r15d
  unsigned __int8 v14; // r15
  unsigned __int64 v15; // rsi
  int v16; // eax
  _BYTE *v17; // rsi
  unsigned __int8 v20; // [rsp+2Eh] [rbp-62h]
  unsigned __int8 v21; // [rsp+2Fh] [rbp-61h]
  _QWORD v22[4]; // [rsp+30h] [rbp-60h] BYREF
  char v23; // [rsp+50h] [rbp-40h]
  char v24; // [rsp+51h] [rbp-3Fh]

  v21 = a4;
  sub_A74A00((__int64)a2);
  v7 = *(_DWORD *)(a1 + 240);
  if ( v7 == 9 )
    return 0;
  v8 = 255;
  v20 = 0;
  while ( v7 == 512 )
  {
    if ( (unsigned __int8)sub_120C270(a1, a2) )
      return 1;
LABEL_25:
    v7 = *(_DWORD *)(a1 + 240);
LABEL_26:
    if ( v7 == 9 )
    {
      v11 = v20;
      goto LABEL_19;
    }
  }
  if ( v7 == 505 )
  {
    if ( v21 )
    {
      v15 = *(_QWORD *)(a1 + 232);
      v24 = 1;
      v22[0] = "cannot have an attribute group reference in an attribute group";
      v23 = 3;
      sub_11FD800(a1 + 176, v15, (__int64)v22, 1);
      v20 = v21;
    }
    else
    {
      v16 = *(_DWORD *)(a1 + 280);
      LODWORD(v22[0]) = v16;
      v17 = *(_BYTE **)(a3 + 8);
      if ( v17 == *(_BYTE **)(a3 + 16) )
      {
        sub_C88AB0(a3, v17, v22);
      }
      else
      {
        if ( v17 )
        {
          *(_DWORD *)v17 = v16;
          v17 = *(_BYTE **)(a3 + 8);
        }
        *(_QWORD *)(a3 + 8) = v17 + 4;
      }
    }
    goto LABEL_33;
  }
  v9 = *(_QWORD *)(a1 + 232);
  if ( v7 == 169 )
  {
    *a5 = v9;
LABEL_23:
    v13 = v7 - 165;
    if ( (unsigned __int8)sub_1217C30(a1, v7 - 165, a2, v21) )
      return 1;
    v14 = (sub_A719F0(v13) ^ 1) & (v13 != 86);
    if ( v14 )
    {
      v22[0] = "this attribute does not apply to functions";
      v24 = 1;
      v23 = 3;
      sub_11FD800(a1 + 176, v9, (__int64)v22, 1);
      v20 = v14;
      v7 = *(_DWORD *)(a1 + 240);
      goto LABEL_26;
    }
    goto LABEL_25;
  }
  if ( v7 == 270 )
  {
    v8 &= 3u;
    goto LABEL_33;
  }
  if ( v7 > 0x10E )
  {
    if ( v7 == 271 )
    {
      v8 &= 0xCu;
    }
    else
    {
      if ( v7 != 272 )
        goto LABEL_18;
      v8 &= 0xFu;
    }
    goto LABEL_33;
  }
  switch ( v7 )
  {
    case 0xD8u:
      v8 &= 0x55u;
      goto LABEL_33;
    case 0xF3u:
      v8 &= 0xAAu;
LABEL_33:
      v7 = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = v7;
      goto LABEL_26;
    case 0xD7u:
      v8 = 0;
      goto LABEL_33;
  }
  if ( v7 - 166 <= 0x61 )
    goto LABEL_23;
LABEL_18:
  v10 = *(_QWORD *)(a1 + 232);
  v11 = v20;
  if ( (_BYTE)a4 )
  {
    v24 = 1;
    v23 = 3;
    v11 = a4;
    v22[0] = "unterminated attribute group";
    sub_11FD800(a1 + 176, v10, (__int64)v22, 1);
    return v11;
  }
LABEL_19:
  if ( v8 != 255 )
    sub_A77CD0(a2, v8);
  return v11;
}
