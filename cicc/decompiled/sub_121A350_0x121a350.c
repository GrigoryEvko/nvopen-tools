// Function: sub_121A350
// Address: 0x121a350
//
__int64 __fastcall sub_121A350(__int64 a1, __int64 **a2, int a3, int a4)
{
  unsigned int v4; // r12d
  __int64 v6; // r15
  int v8; // eax
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rsi
  int v11; // eax
  __int64 *v12; // rdx
  __int64 *v13; // [rsp+18h] [rbp-68h] BYREF
  int v14[8]; // [rsp+20h] [rbp-60h] BYREF
  char v15; // [rsp+40h] [rbp-40h]
  char v16; // [rsp+41h] [rbp-3Fh]

  v13 = 0;
  if ( a3 == *(_DWORD *)(a1 + 240) )
  {
    v6 = a1 + 176;
    v8 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v8;
    if ( v8 != 12 )
    {
      v16 = 1;
      v9 = *(_QWORD *)(a1 + 232);
      v15 = 3;
      v4 = 1;
      *(_QWORD *)v14 = "expected '('";
      sub_11FD800(v6, v9, (__int64)v14, 1);
      return v4;
    }
    *(_DWORD *)(a1 + 240) = sub_1205200(v6);
    v16 = 1;
    *(_QWORD *)v14 = "expected type";
    v15 = 3;
    v4 = sub_12190A0(a1, &v13, v14, 0);
    if ( !(_BYTE)v4 )
    {
      if ( *(_DWORD *)(a1 + 240) == 13 )
      {
        v11 = sub_1205200(v6);
        v12 = v13;
        *(_DWORD *)(a1 + 240) = v11;
        sub_A77E60(a2, a4, (__int64)v12);
        return v4;
      }
      v10 = *(_QWORD *)(a1 + 232);
      v16 = 1;
      v15 = 3;
      *(_QWORD *)v14 = "expected ')'";
      sub_11FD800(v6, v10, (__int64)v14, 1);
    }
  }
  return 1;
}
