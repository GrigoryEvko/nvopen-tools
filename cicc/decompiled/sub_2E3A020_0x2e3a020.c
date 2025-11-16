// Function: sub_2E3A020
// Address: 0x2e3a020
//
__int64 *__fastcall sub_2E3A020(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v4; // [rsp-8h] [rbp-8h]

  v2 = *a1;
  if ( v2 )
    return sub_FE8740(v2, **(_QWORD **)(v2 + 128), a2, 0);
  *((_BYTE *)&v4 - 24) = 0;
  return (__int64 *)*(&v4 - 4);
}
