// Function: sub_388CB60
// Address: 0x388cb60
//
__int64 __fastcall sub_388CB60(__int64 a1, unsigned int *a2, _BYTE *a3)
{
  __int64 v3; // r14
  __int64 result; // rax
  int v6; // eax
  unsigned __int64 v7; // rsi
  const char *v8; // [rsp+0h] [rbp-40h] BYREF
  char v9; // [rsp+10h] [rbp-30h]
  char v10; // [rsp+11h] [rbp-2Fh]

  v3 = a1 + 8;
  *a3 = 0;
  while ( 1 )
  {
    if ( *(_DWORD *)(a1 + 64) != 4 )
      return 0;
    v6 = sub_3887100(v3);
    *(_DWORD *)(a1 + 64) = v6;
    if ( v6 == 376 )
    {
      *a3 = 1;
      return 0;
    }
    if ( v6 != 88 )
      break;
    result = sub_388C5A0(a1, a2);
    if ( (_BYTE)result )
      return result;
  }
  v7 = *(_QWORD *)(a1 + 56);
  v10 = 1;
  v9 = 3;
  v8 = "expected metadata or 'align'";
  return sub_38814C0(v3, v7, (__int64)&v8);
}
