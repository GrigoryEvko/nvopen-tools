// Function: sub_12252E0
// Address: 0x12252e0
//
__int64 __fastcall sub_12252E0(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned int v4; // r8d
  bool v5; // zf
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v15; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v16; // [rsp+28h] [rbp-98h] BYREF
  const char *v17; // [rsp+30h] [rbp-90h] BYREF
  char v18; // [rsp+50h] [rbp-70h]
  char v19; // [rsp+51h] [rbp-6Fh]
  __int64 *v20; // [rsp+60h] [rbp-60h] BYREF
  __int64 v21; // [rsp+68h] [rbp-58h]
  _BYTE v22[80]; // [rsp+70h] [rbp-50h] BYREF

  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v4 = sub_120AFE0(a1, 12, "expected '(' here");
  if ( !(_BYTE)v4 )
  {
    v5 = *(_DWORD *)(a1 + 240) == 13;
    v20 = (__int64 *)v22;
    v21 = 0x400000000LL;
    if ( v5 )
    {
LABEL_9:
      v6 = 13;
      v4 = sub_120AFE0(a1, 13, "expected ')' here");
      if ( !(_BYTE)v4 )
      {
        v6 = (__int64)v20;
        v12 = sub_B00B60(*(__int64 **)a1, v20, (unsigned int)v21);
        v10 = v20;
        v4 = 0;
        *a2 = v12;
        if ( v10 == (__int64 *)v22 )
          return v4;
        goto LABEL_11;
      }
    }
    else
    {
      while ( 1 )
      {
        v6 = (__int64)&v16;
        v19 = 1;
        v17 = "expected value-as-metadata operand";
        v18 = 3;
        v4 = sub_1225220(a1, &v16, (int *)&v17, a3);
        if ( (_BYTE)v4 )
          break;
        v8 = 0;
        v9 = (unsigned int)v21;
        if ( (unsigned int)*v16 - 1 <= 1 )
          v8 = (__int64)v16;
        if ( (unsigned __int64)(unsigned int)v21 + 1 > HIDWORD(v21) )
        {
          v13 = v8;
          sub_C8D5F0((__int64)&v20, v22, (unsigned int)v21 + 1LL, 8u, v8, v7);
          v9 = (unsigned int)v21;
          v8 = v13;
        }
        v20[v9] = v8;
        LODWORD(v21) = v21 + 1;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          goto LABEL_9;
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      }
    }
    v10 = v20;
    if ( v20 != (__int64 *)v22 )
    {
LABEL_11:
      v15 = v4;
      _libc_free(v10, v6);
      return v15;
    }
  }
  return v4;
}
