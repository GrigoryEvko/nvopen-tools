// Function: sub_EB9680
// Address: 0xeb9680
//
__int64 __fastcall sub_EB9680(__int64 a1)
{
  unsigned __int8 v1; // bl
  __int64 result; // rax
  char v3; // al
  __int64 v4; // rsi
  __int64 v5; // rdx
  unsigned __int8 v6; // [rsp+Fh] [rbp-71h]
  __int64 v7; // [rsp+10h] [rbp-70h] BYREF
  __int64 v8; // [rsp+18h] [rbp-68h]
  _QWORD v9[4]; // [rsp+20h] [rbp-60h] BYREF
  char v10; // [rsp+40h] [rbp-40h]
  char v11; // [rsp+41h] [rbp-3Fh]

  v7 = 0;
  v8 = 0;
  if ( (unsigned __int8)sub_ECE2A0(a1, 9) )
  {
    v4 = 0;
    v5 = 0;
  }
  else
  {
    v6 = 0;
    v1 = 0;
    while ( 1 )
    {
      if ( (unsigned __int8)sub_EB61F0(a1, &v7) )
      {
        v11 = 1;
        v9[0] = "expected .eh_frame or .debug_frame";
        v10 = 3;
        return sub_ECE0E0(a1, v9, 0, 0);
      }
      if ( v8 == 9 )
      {
        if ( *(_QWORD *)v7 == 0x6D6172665F68652ELL && *(_BYTE *)(v7 + 8) == 101 )
          v1 = 1;
      }
      else if ( v8 == 12 && *(_QWORD *)v7 == 0x665F67756265642ELL )
      {
        v3 = v6;
        if ( *(_DWORD *)(v7 + 8) == 1701667186 )
          v3 = 1;
        v6 = v3;
      }
      if ( (unsigned __int8)sub_ECE2A0(a1, 9) )
        break;
      v11 = 1;
      v9[0] = "expected comma";
      v10 = 3;
      result = sub_ECE210(a1, 26, v9);
      if ( (_BYTE)result )
        return result;
    }
    v5 = v6;
    v4 = v1;
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 232) + 856LL))(*(_QWORD *)(a1 + 232), v4, v5);
  return 0;
}
