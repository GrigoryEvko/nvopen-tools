// Function: sub_1212970
// Address: 0x1212970
//
__int64 __fastcall sub_1212970(__int64 a1, _DWORD *a2, unsigned __int64 *a3, _BYTE *a4)
{
  __int64 v4; // r15
  __int64 result; // rax
  int v9; // eax
  unsigned __int64 v10; // rsi
  const char *v11; // [rsp+0h] [rbp-60h] BYREF
  char v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+21h] [rbp-3Fh]

  v4 = a1 + 176;
  *a4 = 0;
  while ( 1 )
  {
    if ( *(_DWORD *)(a1 + 240) != 4 )
      return 0;
    v9 = sub_1205200(v4);
    *(_DWORD *)(a1 + 240) = v9;
    if ( v9 == 511 )
    {
      *a4 = 1;
      return 0;
    }
    v10 = *(_QWORD *)(a1 + 232);
    *a3 = v10;
    if ( *(_DWORD *)(a1 + 240) != 94 )
      break;
    result = sub_1212650(a1, a2, 0);
    if ( (_BYTE)result )
      return result;
  }
  v13 = 1;
  v12 = 3;
  v11 = "expected metadata or 'addrspace'";
  sub_11FD800(v4, v10, (__int64)&v11, 1);
  return 1;
}
