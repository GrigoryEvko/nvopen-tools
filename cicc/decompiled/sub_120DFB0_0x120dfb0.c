// Function: sub_120DFB0
// Address: 0x120dfb0
//
__int64 __fastcall sub_120DFB0(__int64 a1, _DWORD *a2, __int64 a3)
{
  __int64 v3; // r14
  int v6; // eax
  unsigned __int64 v7; // rsi
  const char *v8; // rax
  unsigned int v9; // r15d
  int v11; // eax
  unsigned __int64 v12; // [rsp+8h] [rbp-78h]
  int v13; // [rsp+1Ch] [rbp-64h] BYREF
  _QWORD v14[4]; // [rsp+20h] [rbp-60h] BYREF
  char v15; // [rsp+40h] [rbp-40h]
  char v16; // [rsp+41h] [rbp-3Fh]

  v3 = a1 + 176;
  v6 = sub_1205200(a1 + 176);
  v7 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = v6;
  if ( v6 == 12 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(v3);
    v9 = sub_120BD00(a1, a2);
    if ( (_BYTE)v9 )
      return v9;
    v11 = *(_DWORD *)(a1 + 240);
    if ( v11 == 4 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v3);
      v12 = *(_QWORD *)(a1 + 232);
      if ( (unsigned __int8)sub_120BD00(a1, &v13) )
        return 1;
      if ( *a2 == v13 )
      {
        v16 = 1;
        v14[0] = "'allocsize' indices can't refer to the same parameter";
        v15 = 3;
        sub_11FD800(v3, v12, (__int64)v14, 1);
        return 1;
      }
      *(_DWORD *)a3 = v13;
      *(_BYTE *)(a3 + 4) = 1;
      v11 = *(_DWORD *)(a1 + 240);
    }
    else if ( *(_BYTE *)(a3 + 4) )
    {
      *(_BYTE *)(a3 + 4) = 0;
      v11 = *(_DWORD *)(a1 + 240);
    }
    v7 = *(_QWORD *)(a1 + 232);
    if ( v11 == 13 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v3);
      return v9;
    }
    v16 = 1;
    v8 = "expected ')'";
  }
  else
  {
    v16 = 1;
    v8 = "expected '('";
  }
  v14[0] = v8;
  v15 = 3;
  sub_11FD800(v3, v7, (__int64)v14, 1);
  return 1;
}
