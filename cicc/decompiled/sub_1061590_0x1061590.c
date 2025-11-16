// Function: sub_1061590
// Address: 0x1061590
//
__int64 *__fastcall sub_1061590(__int64 *a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // r14
  unsigned int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // rbx

  v2 = sub_C63BB0();
  v4 = v3;
  v5 = v2;
  v6 = sub_22077B0(64);
  v7 = v6;
  if ( v6 )
    sub_C63EB0(v6, a2, v5, v4);
  *a1 = v7 | 1;
  return a1;
}
