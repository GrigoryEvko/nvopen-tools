// Function: sub_C3BD20
// Address: 0xc3bd20
//
__int64 __fastcall sub_C3BD20(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r12d
  int v4; // ebx
  int v5; // r12d
  _QWORD v6[2]; // [rsp+0h] [rbp-40h] BYREF
  int v7; // [rsp+10h] [rbp-30h]

  v1 = *(_BYTE *)(a1 + 20) & 7;
  if ( v1 == 1 )
    return 0x80000000;
  if ( v1 == 3 )
    return (unsigned int)-2147483647;
  v2 = 0x7FFFFFFF;
  if ( !v1 )
    return v2;
  if ( !sub_C33940(a1) )
    return *(unsigned int *)(a1 + 16);
  sub_C33EB0(v6, (__int64 *)a1);
  v4 = *(_DWORD *)(*(_QWORD *)a1 + 8LL) - 1;
  v7 += v4;
  sub_C36450((__int64)v6, 1, 0);
  v5 = v7;
  sub_C338F0((__int64)v6);
  return (unsigned int)(v5 - v4);
}
