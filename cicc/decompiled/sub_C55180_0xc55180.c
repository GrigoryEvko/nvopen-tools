// Function: sub_C55180
// Address: 0xc55180
//
__int64 __fastcall sub_C55180(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 result; // rax
  unsigned __int8 v11; // al
  char **v12; // rbx
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // [rsp-8h] [rbp-50h]
  unsigned __int8 v16; // [rsp+17h] [rbp-31h]
  _BYTE v17[33]; // [rsp+27h] [rbp-21h] BYREF

  v8 = a1 + 152;
  v9 = a1;
  v17[0] = 0;
  result = sub_C54F80(v8, a1, a3, a4, a5, a6, v17);
  if ( !(_BYTE)result )
  {
    v11 = v17[0];
    if ( v17[0] )
    {
      v12 = *(char ***)(a1 + 136);
      v13 = sub_C4F9D0(v8, v9);
      if ( (unsigned int)(*(_DWORD *)(v13 + 140) - *(_DWORD *)(v13 + 144)) > 1 )
      {
        if ( !qword_4F83C80 )
          sub_C7D570(&qword_4F83C80, sub_C58C10, sub_C51550);
        *(_BYTE *)(qword_4F83C80 + 140) &= 0x9Fu;
        sub_C525B0(v12[1]);
        exit(0);
      }
      sub_C525B0(*v12);
      exit(0);
    }
    v14 = *(_QWORD *)(a1 + 176) == 0;
    *(_WORD *)(a1 + 14) = a2;
    if ( v14 )
      sub_4263D6(v8, a1, v15);
    v16 = v11;
    (*(void (__fastcall **)(__int64, _BYTE *, __int64))(a1 + 184))(a1 + 160, v17, v15);
    return v16;
  }
  return result;
}
