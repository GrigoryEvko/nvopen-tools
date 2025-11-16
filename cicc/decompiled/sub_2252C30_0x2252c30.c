// Function: sub_2252C30
// Address: 0x2252c30
//
__int64 __fastcall sub_2252C30(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v5; // r13
  char *v6; // r13
  __int64 v7; // rsi
  int v8; // ecx
  char v9; // dl
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 result; // rax
  _QWORD *v13; // [rsp+8h] [rbp-30h] BYREF

  v5 = *(_QWORD *)(a1 + 24);
  v13 = a3;
  v6 = (char *)(~a4 + v5);
  while ( 1 )
  {
    v7 = 0;
    v8 = 0;
    do
    {
      v9 = *v6++;
      v10 = (unsigned __int64)(v9 & 0x7F) << v8;
      v8 += 7;
      v7 |= v10;
    }
    while ( v9 < 0 );
    if ( !v7 )
      break;
    v11 = sub_2252B90(a1, v7);
    result = sub_22529E0(v11, a2, &v13);
    if ( (_BYTE)result )
      return result;
  }
  return 0;
}
