// Function: sub_CB56E0
// Address: 0xcb56e0
//
__int64 __fastcall sub_CB56E0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, unsigned int a6)
{
  int v6; // r10d
  int v8; // ecx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned int v16; // [rsp+1Ch] [rbp-44h] BYREF
  _QWORD v17[4]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v18; // [rsp+40h] [rbp-20h]

  v6 = a4;
  if ( a2 == 1 && *a1 == 45 )
  {
    v12 = sub_2241E40(a1, 1, a5, a4, a6);
    *(_DWORD *)a3 = 0;
    *(_QWORD *)(a3 + 8) = v12;
    sub_C87950(a6, 1, v13, v14, a6);
    return 1;
  }
  else
  {
    v17[1] = a2;
    v18 = 261;
    v8 = 3;
    v17[0] = a1;
    if ( (a5 & 1) == 0 )
      v8 = 2;
    v9 = sub_C83360((__int64)v17, (int *)&v16, v6, v8, a6, 0x1B6u);
    *(_DWORD *)a3 = v9;
    *(_QWORD *)(a3 + 8) = v10;
    if ( v9 )
      return 0xFFFFFFFFLL;
    else
      return v16;
  }
}
