// Function: sub_2A72BA0
// Address: 0x2a72ba0
//
__int64 __fastcall sub_2A72BA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 result; // rax
  __int64 v9; // r15
  char v10; // bl
  __int64 v11; // [rsp+8h] [rbp-38h]

  v6 = a2 + 24;
  v11 = a2;
  v7 = *a1;
  do
  {
    sub_2A72020(v7, a2, a3, a4, a5, a6);
    result = v11;
    v9 = *(_QWORD *)(v11 + 32);
    if ( v9 == v6 )
      break;
    v10 = 0;
    do
    {
      a2 = v9 - 56;
      if ( !v9 )
        a2 = 0;
      result = sub_2A6CD80(v7, a2, a3, a4);
      v9 = *(_QWORD *)(v9 + 8);
      v10 |= result;
    }
    while ( v9 != v6 );
  }
  while ( v10 );
  return result;
}
