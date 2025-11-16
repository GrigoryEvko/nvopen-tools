// Function: sub_1514BE0
// Address: 0x1514be0
//
__int64 *__fastcall sub_1514BE0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx

  v2 = sub_14EE0B0();
  v3 = sub_22077B0(56);
  v4 = v3;
  if ( v3 )
    sub_16BCC70(v3, a2, 1, v2);
  *a1 = v4 | 1;
  return a1;
}
