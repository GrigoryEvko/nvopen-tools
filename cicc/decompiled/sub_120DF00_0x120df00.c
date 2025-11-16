// Function: sub_120DF00
// Address: 0x120df00
//
__int64 __fastcall sub_120DF00(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v3; // r14
  __int64 result; // rax
  int v6; // eax
  unsigned __int64 v7; // rsi
  const char *v8; // [rsp+0h] [rbp-50h] BYREF
  char v9; // [rsp+20h] [rbp-30h]
  char v10; // [rsp+21h] [rbp-2Fh]

  v3 = a1 + 176;
  *a3 = 0;
  while ( 1 )
  {
    if ( *(_DWORD *)(a1 + 240) != 4 )
      return 0;
    v6 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v6;
    if ( v6 == 511 )
    {
      *a3 = 1;
      return 0;
    }
    if ( v6 != 251 )
      break;
    result = sub_120CD10(a1, a2, 0);
    if ( (_BYTE)result )
      return result;
  }
  v10 = 1;
  v7 = *(_QWORD *)(a1 + 232);
  v9 = 3;
  v8 = "expected metadata or 'align'";
  sub_11FD800(v3, v7, (__int64)&v8, 1);
  return 1;
}
