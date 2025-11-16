// Function: sub_1252E30
// Address: 0x1252e30
//
void *__fastcall sub_1252E30(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rsi

  *a1 = (__int64)off_49E66F0;
  v1 = a1[2];
  v2 = a1[4];
  v3 = a1[7] + v1 - v2;
  if ( v3 )
  {
    sub_CB6C70((__int64)a1, v3);
    v2 = a1[4];
    v1 = a1[2];
  }
  if ( v2 != v1 )
    sub_CB5AE0(a1);
  return sub_CB5840((__int64)a1);
}
