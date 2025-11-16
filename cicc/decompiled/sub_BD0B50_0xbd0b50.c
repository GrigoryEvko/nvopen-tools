// Function: sub_BD0B50
// Address: 0xbd0b50
//
unsigned __int64 *__fastcall sub_BD0B50(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  unsigned __int64 *result; // rax
  unsigned __int64 v5; // [rsp+8h] [rbp-8h] BYREF

  result = sub_BD0A20(&v5, a1, a2, a3, a4);
  if ( (v5 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  return result;
}
