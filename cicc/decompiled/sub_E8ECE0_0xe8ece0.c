// Function: sub_E8ECE0
// Address: 0xe8ece0
//
char __fastcall sub_E8ECE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  char result; // al
  __int64 v6; // rbx
  __int64 v7; // r14
  _QWORD *v8; // r13
  bool (__fastcall *v9)(__int64, __int64, __int64, __int64); // r15
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  unsigned __int8 v12; // [rsp-44h] [rbp-44h]

  result = 0;
  if ( (*(_DWORD *)a3 & 0xFFFF00) == 0 && (*(_DWORD *)a4 & 0xFFFF00) == 0 )
  {
    v6 = *(_QWORD *)(a4 + 16);
    v7 = *(_QWORD *)(a3 + 16);
    v8 = *(_QWORD **)v6;
    v9 = *(bool (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 40LL);
    if ( !*(_QWORD *)v6 && (*(_BYTE *)(v6 + 9) & 0x70) == 0x20 && *(char *)(v6 + 8) >= 0 )
    {
      *(_BYTE *)(v6 + 8) |= 8u;
      v12 = a5;
      v11 = sub_E807D0(*(_QWORD *)(v6 + 24));
      a5 = v12;
      *(_QWORD *)v6 = v11;
      v8 = v11;
    }
    if ( v9 == sub_E8EB20 )
    {
      v10 = *(_QWORD **)v7;
      if ( !*(_QWORD *)v7 )
      {
        if ( (*(_BYTE *)(v7 + 9) & 0x70) != 0x20 || *(char *)(v7 + 8) < 0 )
          BUG();
        *(_BYTE *)(v7 + 8) |= 8u;
        v10 = sub_E807D0(*(_QWORD *)(v7 + 24));
        *(_QWORD *)v7 = v10;
      }
      return v10[1] == v8[1];
    }
    else
    {
      return ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD *, _QWORD, _QWORD))v9)(a1, a2, v7, v8, a5, 0);
    }
  }
  return result;
}
