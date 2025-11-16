// Function: sub_259C930
// Address: 0x259c930
//
__int64 __fastcall sub_259C930(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v3; // r14
  unsigned __int64 *v4; // rbx
  unsigned __int64 v5; // rsi
  __int64 result; // rax
  char v7; // [rsp+Fh] [rbp-41h] BYREF
  __m128i v8[4]; // [rsp+10h] [rbp-40h] BYREF

  v3 = &a2[a3];
  if ( v3 == a2 )
    return 1;
  v4 = a2;
  while ( 1 )
  {
    v5 = *v4;
    if ( **(_BYTE **)a1 == 3 )
      sub_250D230((unsigned __int64 *)v8, v5, 2, 0);
    else
      sub_250D230((unsigned __int64 *)v8, v5, 4, 0);
    result = sub_259B8C0(*(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24), v8, 0, &v7, 0, 0);
    if ( !(_BYTE)result )
      break;
    if ( v3 == ++v4 )
      return 1;
  }
  return result;
}
