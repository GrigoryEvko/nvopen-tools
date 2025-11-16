// Function: sub_120CEA0
// Address: 0x120cea0
//
__int64 __fastcall sub_120CEA0(__int64 a1, _DWORD *a2)
{
  int v3; // eax
  _BYTE *v4; // rsi
  __int64 v5; // rdx
  unsigned int v6; // r12d
  unsigned __int64 v8; // rsi
  __int64 v9[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-70h] BYREF
  const char *v11; // [rsp+20h] [rbp-60h] BYREF
  char v12; // [rsp+40h] [rbp-40h]
  char v13; // [rsp+41h] [rbp-3Fh]

  v3 = sub_1205200(a1 + 176);
  v4 = *(_BYTE **)(a1 + 248);
  v9[0] = (__int64)v10;
  v5 = *(_QWORD *)(a1 + 256);
  *(_DWORD *)(a1 + 240) = v3;
  sub_12060D0(v9, v4, (__int64)&v4[v5]);
  if ( !(unsigned int)sub_2241AC0(v9, "tiny") )
  {
    *a2 = 0;
LABEL_3:
    v6 = sub_120AFE0(a1, 512, "expected global code model string");
    goto LABEL_4;
  }
  if ( !(unsigned int)sub_2241AC0(v9, "small") )
  {
    *a2 = 1;
    goto LABEL_3;
  }
  if ( !(unsigned int)sub_2241AC0(v9, "kernel") )
  {
    *a2 = 2;
    goto LABEL_3;
  }
  if ( !(unsigned int)sub_2241AC0(v9, "medium") )
  {
    *a2 = 3;
    goto LABEL_3;
  }
  if ( !(unsigned int)sub_2241AC0(v9, "large") )
  {
    *a2 = 4;
    goto LABEL_3;
  }
  v13 = 1;
  v8 = *(_QWORD *)(a1 + 232);
  v12 = 3;
  v6 = 1;
  v11 = "expected global code model string";
  sub_11FD800(a1 + 176, v8, (__int64)&v11, 1);
LABEL_4:
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  return v6;
}
