// Function: sub_120E760
// Address: 0x120e760
//
__int64 __fastcall sub_120E760(__int64 a1, __int64 **a2)
{
  __int64 v2; // r13
  unsigned int v3; // eax
  unsigned int v4; // r12d
  int v5; // eax
  __int16 *v6; // r14
  const char *v7; // rax
  unsigned __int64 v8; // rsi
  int v10; // eax
  int v11; // eax
  bool v12; // zf
  char v13; // dl
  __int16 v14; // cx
  int v15; // eax
  unsigned __int64 v16; // rsi
  char v17; // [rsp+Dh] [rbp-63h] BYREF
  __int16 v18; // [rsp+Eh] [rbp-62h] BYREF
  _QWORD v19[4]; // [rsp+10h] [rbp-60h] BYREF
  char v20; // [rsp+30h] [rbp-40h]
  char v21; // [rsp+31h] [rbp-3Fh]

  v2 = a1 + 176;
  *(_BYTE *)(a1 + 336) = 1;
  v18 = 0;
  v17 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v3 = sub_120AFE0(a1, 12, "expected '('");
  if ( (_BYTE)v3 )
  {
LABEL_10:
    v4 = 1;
    goto LABEL_11;
  }
  v4 = v3;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 == 375 )
    goto LABEL_12;
  v6 = (__int16 *)&v17;
  if ( v5 == 55 )
    goto LABEL_15;
LABEL_4:
  switch ( v5 )
  {
    case 275:
      v15 = sub_1205200(v2);
      *(_BYTE *)v6 |= 1u;
      *(_DWORD *)(a1 + 240) = v15;
      break;
    case 274:
      v15 = sub_1205200(v2);
      *(_BYTE *)v6 |= 3u;
      *(_DWORD *)(a1 + 240) = v15;
      break;
    case 276:
      v15 = sub_1205200(v2);
      *(_BYTE *)v6 |= 0xCu;
      *(_DWORD *)(a1 + 240) = v15;
      break;
    case 277:
      v15 = sub_1205200(v2);
      *(_BYTE *)v6 |= 4u;
      *(_DWORD *)(a1 + 240) = v15;
      break;
    default:
      v21 = 1;
      v7 = "expected one of 'none', 'address', 'address_is_null', 'provenance' or 'read_provenance'";
LABEL_9:
      v19[0] = v7;
      v8 = *(_QWORD *)(a1 + 232);
      v20 = 3;
      sub_11FD800(v2, v8, (__int64)v19, 1);
      goto LABEL_10;
  }
  if ( v15 != 13 )
  {
    while ( !(unsigned __int8)sub_120AFE0(a1, 4, "expected ',' or ')'") )
    {
      v5 = *(_DWORD *)(a1 + 240);
      if ( v5 != 375 )
      {
        if ( v5 == 55 )
        {
          *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        }
        else if ( *(_BYTE *)v6 )
        {
          goto LABEL_4;
        }
        v21 = 1;
        v7 = "cannot use 'none' with other component";
        goto LABEL_9;
      }
LABEL_12:
      *(_DWORD *)(a1 + 240) = sub_1205200(v2);
      if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") )
        goto LABEL_10;
      if ( HIBYTE(v18) )
      {
        v21 = 1;
        v16 = *(_QWORD *)(a1 + 232);
        v20 = 3;
        v4 = HIBYTE(v18);
        v19[0] = "duplicate 'ret' location";
        sub_11FD800(v2, v16, (__int64)v19, 1);
        goto LABEL_11;
      }
      v6 = &v18;
      v18 = 256;
      v5 = *(_DWORD *)(a1 + 240);
      if ( v5 != 55 )
        goto LABEL_4;
LABEL_15:
      v10 = sub_1205200(v2);
      *(_BYTE *)v6 = 0;
      *(_DWORD *)(a1 + 240) = v10;
      if ( v10 == 13 )
        goto LABEL_16;
    }
    goto LABEL_10;
  }
LABEL_16:
  v11 = sub_1205200(v2);
  v12 = HIBYTE(v18) == 0;
  *(_DWORD *)(a1 + 240) = v11;
  v13 = v17;
  if ( !v12 )
    v13 = v18;
  LOBYTE(v14) = v17;
  HIBYTE(v14) = v13;
  sub_A77CE0(a2, v14);
LABEL_11:
  *(_BYTE *)(a1 + 336) = 0;
  return v4;
}
