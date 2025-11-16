// Function: sub_D9AF00
// Address: 0xd9af00
//
char __fastcall sub_D9AF00(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rbx
  __int64 *v6; // r14
  char result; // al
  __int64 v8; // rsi

  v5 = a2;
  v6 = &a2[a3];
  sub_D9AB00((__int64)a1, 0, 0, 0);
  *a1 = &unk_49DEA40;
  a1[5] = a1 + 7;
  result = 0;
  a1[6] = 0x1000000000LL;
  if ( a2 != v6 )
  {
    do
    {
      v8 = *v5++;
      result = sub_D9AD80((__int64)a1, v8, a4);
    }
    while ( v6 != v5 );
  }
  return result;
}
