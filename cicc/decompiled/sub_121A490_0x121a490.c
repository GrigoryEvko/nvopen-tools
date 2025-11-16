// Function: sub_121A490
// Address: 0x121a490
//
__int64 __fastcall sub_121A490(__int64 a1, _QWORD *a2, unsigned int a3)
{
  __int64 v3; // r15
  char v4; // r14
  int v6; // eax
  unsigned __int64 v7; // rsi
  unsigned int v8; // r12d
  unsigned __int8 v10; // r11
  unsigned __int64 v11; // rax
  const char *v12; // rax
  unsigned __int64 v13; // [rsp+8h] [rbp-98h]
  __int64 v14; // [rsp+18h] [rbp-88h]
  unsigned __int64 v15; // [rsp+20h] [rbp-80h]
  unsigned __int8 v16; // [rsp+20h] [rbp-80h]
  unsigned __int8 v17; // [rsp+28h] [rbp-78h]
  unsigned __int64 v18; // [rsp+28h] [rbp-78h]
  __int64 *v19; // [rsp+38h] [rbp-68h] BYREF
  __int64 v20[4]; // [rsp+40h] [rbp-60h] BYREF
  char v21; // [rsp+60h] [rbp-40h]
  char v22; // [rsp+61h] [rbp-3Fh]

  v3 = a1 + 176;
  v4 = 0;
  v6 = *(_DWORD *)(a1 + 240);
  if ( (_BYTE)a3 && v6 == 18 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( (unsigned __int8)sub_120AFE0(a1, 19, "expected 'x' after vscale") )
      return 1;
    v6 = *(_DWORD *)(a1 + 240);
    v4 = a3;
  }
  if ( v6 != 529 || !*(_BYTE *)(a1 + 332) || *(_DWORD *)(a1 + 328) > 0x40u )
  {
    v7 = *(_QWORD *)(a1 + 232);
    v22 = 1;
    v20[0] = (__int64)"expected number in address space";
    v21 = 3;
    sub_11FD800(v3, v7, (__int64)v20, 1);
    return 1;
  }
  v15 = *(_QWORD *)(a1 + 232);
  v17 = *(_BYTE *)(a1 + 332);
  v14 = *(_QWORD *)(a1 + 320);
  *(_DWORD *)(a1 + 240) = sub_1205200(v3);
  if ( (unsigned __int8)sub_120AFE0(a1, 19, "expected 'x' after element count") )
    return 1;
  v10 = v17;
  v11 = *(_QWORD *)(a1 + 232);
  v19 = 0;
  v18 = v15;
  v13 = v11;
  v20[0] = (__int64)"expected type";
  v16 = v10;
  v22 = 1;
  v21 = 3;
  if ( (unsigned __int8)sub_12190A0(a1, &v19, (int *)v20, 0) )
    return 1;
  if ( (_BYTE)a3 )
  {
    v8 = sub_120AFE0(a1, 11, "expected end of sequential type");
    if ( (_BYTE)v8 )
      return 1;
    if ( !v14 )
    {
      v22 = 1;
      v12 = "zero element vector is illegal";
      goto LABEL_18;
    }
    if ( v14 != (unsigned int)v14 )
    {
      v22 = 1;
      v12 = "size too large for vector";
LABEL_18:
      v8 = a3;
      v20[0] = (__int64)v12;
      v21 = 3;
      sub_11FD800(v3, v18, (__int64)v20, 1);
      return v8;
    }
    if ( (unsigned __int8)sub_BCBCB0((__int64)v19) )
    {
      BYTE4(v20[0]) = v4;
      LODWORD(v20[0]) = v14;
      *a2 = sub_BCE1B0(v19, v20[0]);
    }
    else
    {
      v22 = 1;
      v8 = a3;
      v21 = 3;
      v20[0] = (__int64)"invalid vector element type";
      sub_11FD800(v3, v13, (__int64)v20, 1);
    }
  }
  else
  {
    v8 = sub_120AFE0(a1, 7, "expected end of sequential type");
    if ( (_BYTE)v8 )
      return 1;
    if ( (unsigned __int8)sub_BCBC60((__int64)v19) )
    {
      *a2 = sub_BCD420(v19, v14);
    }
    else
    {
      v22 = 1;
      v20[0] = (__int64)"invalid array element type";
      v21 = 3;
      sub_11FD800(v3, v13, (__int64)v20, 1);
      return v16;
    }
  }
  return v8;
}
