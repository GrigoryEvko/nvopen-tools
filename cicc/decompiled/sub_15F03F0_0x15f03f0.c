// Function: sub_15F03F0
// Address: 0x15f03f0
//
__int64 *__fastcall sub_15F03F0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r14
  unsigned __int64 v5; // r15
  __int64 *v7; // [rsp+8h] [rbp-38h]

  v2 = a1;
  v3 = *a1;
  while ( 1 )
  {
    v4 = *(v2 - 1);
    v7 = v2--;
    v5 = *(_QWORD *)(sub_15EFCB0(a2, v3) + 784);
    if ( v5 <= *(_QWORD *)(sub_15EFCB0(a2, v4) + 784) )
      break;
    v2[1] = *v2;
  }
  *v7 = v3;
  return v7;
}
