// Function: sub_B1C840
// Address: 0xb1c840
//
__int64 __fastcall sub_B1C840(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r14
  unsigned int v3; // ebx
  int v4; // r12d
  __int64 v5; // r15
  __int64 result; // rax
  unsigned __int64 v7; // [rsp+8h] [rbp-38h]

  v2 = a1 + 2;
  a1[1] = 0x800000000LL;
  *a1 = a1 + 2;
  v3 = *((_DWORD *)a2 + 2);
  v4 = *((_DWORD *)a2 + 6);
  v5 = *a2;
  v7 = (int)(v3 - v4);
  result = 0;
  if ( v7 > 8 )
  {
    sub_C8D5F0(a1, a1 + 2, v7, 8);
    result = *((unsigned int *)a1 + 2);
    v2 = (_QWORD *)(*a1 + 8 * result);
  }
  if ( v3 != v4 )
  {
    do
    {
      --v3;
      if ( v2 )
        *v2 = sub_B46EC0(v5, v3);
      ++v2;
    }
    while ( v3 != v4 );
    result = *((unsigned int *)a1 + 2);
  }
  *((_DWORD *)a1 + 2) = result + v7;
  return result;
}
