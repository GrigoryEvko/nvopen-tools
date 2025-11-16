// Function: sub_3950AD0
// Address: 0x3950ad0
//
__int64 *__fastcall sub_3950AD0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 i; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+10h] [rbp-30h]
  __int64 v12; // [rsp+18h] [rbp-28h]

  v4 = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
  for ( i = v4; v4; i = v4 )
  {
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v4) + 16) - 25) <= 9u )
      break;
    v4 = *(_QWORD *)(v4 + 8);
  }
  v10 = a2;
  v12 = a3;
  v11 = 0;
  sub_3950920(&i);
  v5 = i;
  a1[5] = a2;
  a1[4] = 0;
  *a1 = v5;
  v6 = v10;
  a1[6] = 0;
  a1[1] = v6;
  v7 = v11;
  a1[7] = 0;
  a1[2] = v7;
  a1[3] = v12;
  return a1;
}
