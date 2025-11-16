// Function: sub_101C7F0
// Address: 0x101c7f0
//
unsigned __int8 *__fastcall sub_101C7F0(int a1, __int64 *a2, unsigned __int8 *a3, unsigned int a4, __m128i *a5, int a6)
{
  unsigned int v7; // ebx
  int v8; // eax
  unsigned __int8 *result; // rax
  int v10; // eax
  __m128i *v11; // [rsp-30h] [rbp-30h]
  unsigned int v12; // [rsp-20h] [rbp-20h]

  if ( !a6 )
    return 0;
  v7 = a6 - 1;
  v8 = *(unsigned __int8 *)a2;
  if ( (unsigned __int8)v8 <= 0x1Cu
    || (unsigned int)(v8 - 42) > 0x11
    || a4 != v8 - 29
    || (v11 = a5,
        v12 = a4,
        result = sub_101C660(a1, (unsigned __int8 *)a2, (__int64 *)a3, a4, a5, v7),
        a4 = v12,
        a5 = v11,
        !result) )
  {
    v10 = *a3;
    if ( (unsigned __int8)v10 > 0x1Cu && (unsigned int)(v10 - 42) <= 0x11 && a4 == v10 - 29 )
      return sub_101C660(a1, a3, a2, a4, a5, v7);
    else
      return 0;
  }
  return result;
}
