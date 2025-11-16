// Function: sub_388CBF0
// Address: 0x388cbf0
//
__int64 __fastcall sub_388CBF0(__int64 a1, _DWORD *a2, unsigned __int64 *a3, _BYTE *a4)
{
  __int64 v4; // r15
  __int64 result; // rax
  int v9; // eax
  unsigned __int64 v10; // rsi
  const char *v11; // [rsp+0h] [rbp-50h] BYREF
  char v12; // [rsp+10h] [rbp-40h]
  char v13; // [rsp+11h] [rbp-3Fh]

  v4 = a1 + 8;
  *a4 = 0;
  while ( 1 )
  {
    if ( *(_DWORD *)(a1 + 64) != 4 )
      return 0;
    v9 = sub_3887100(v4);
    *(_DWORD *)(a1 + 64) = v9;
    if ( v9 == 376 )
    {
      *a4 = 1;
      return 0;
    }
    v10 = *(_QWORD *)(a1 + 56);
    *a3 = v10;
    if ( *(_DWORD *)(a1 + 64) != 89 )
      break;
    result = sub_388BF60(a1, a2);
    if ( (_BYTE)result )
      return result;
  }
  v13 = 1;
  v11 = "expected metadata or 'addrspace'";
  v12 = 3;
  return sub_38814C0(v4, v10, (__int64)&v11);
}
