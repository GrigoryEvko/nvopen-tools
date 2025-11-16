// Function: sub_1249310
// Address: 0x1249310
//
__int64 __fastcall sub_1249310(
        __int64 a1,
        unsigned __int8 a2,
        void (__fastcall *a3)(_QWORD *, __int64, _QWORD, _QWORD, _BYTE *, __int64),
        __int64 a4)
{
  __int64 v6; // r13
  int v8; // eax
  __int64 v9; // rdi
  unsigned __int8 v10; // al
  unsigned __int64 v12; // rsi
  unsigned __int8 v13; // [rsp+Fh] [rbp-61h]
  const char *v14; // [rsp+10h] [rbp-60h] BYREF
  char v15; // [rsp+30h] [rbp-40h]
  char v16; // [rsp+31h] [rbp-3Fh]

  v6 = a1 + 176;
  v8 = sub_1205200(a1 + 176);
  v9 = *(_QWORD *)a1;
  *(_DWORD *)(a1 + 240) = v8;
  v10 = sub_B6F8E0(v9);
  if ( v10 )
  {
    v13 = v10;
    v12 = *(_QWORD *)(a1 + 232);
    v16 = 1;
    v14 = "Can't read textual IR with a Context that discards named Values";
    v15 = 3;
    sub_11FD800(v6, v12, (__int64)&v14, 1);
    return v13;
  }
  else if ( *(_QWORD *)(a1 + 344) && (unsigned __int8)sub_1216280(a1, a3, a4)
         || (unsigned __int8)sub_1249060(a1)
         || (unsigned __int8)sub_1214F10((_QWORD *)a1, a2) )
  {
    return 1;
  }
  else
  {
    return sub_120A540((_QWORD *)a1);
  }
}
