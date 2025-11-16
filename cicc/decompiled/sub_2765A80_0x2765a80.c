// Function: sub_2765A80
// Address: 0x2765a80
//
bool __fastcall sub_2765A80(__int64 *a1)
{
  __int64 *v1; // rbx
  __int64 v2; // r14
  __int64 v3; // r12
  bool result; // al
  __int64 v5; // rsi
  __int64 *v6; // r13
  __int64 v7; // rsi

  v1 = a1;
  v2 = *a1;
  v3 = a1[1];
  while ( 1 )
  {
    v5 = *(v1 - 2);
    v6 = v1;
    if ( v2 == v5 )
      break;
    v1 -= 2;
    result = sub_B445A0(v2, v5);
    if ( !result )
      goto LABEL_6;
LABEL_3:
    v1[2] = *v1;
    v1[3] = v1[1];
  }
  v7 = *(v1 - 1);
  v1 -= 2;
  result = sub_B445A0(v3, v7);
  if ( result )
    goto LABEL_3;
LABEL_6:
  *v6 = v2;
  v6[1] = v3;
  return result;
}
