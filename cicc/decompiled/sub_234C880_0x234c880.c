// Function: sub_234C880
// Address: 0x234c880
//
__int64 *__fastcall sub_234C880(__int64 *a1, __int64 *a2)
{
  char v2; // al
  __int64 v3; // rax
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  v2 = *((_BYTE *)a2 + 24);
  *((_BYTE *)a2 + 24) = v2 & 0xFD;
  if ( (v2 & 1) != 0 )
  {
    v3 = *a2;
    *a2 = 0;
    *a1 = v3 | 1;
  }
  else
  {
    *a1 = 1;
    v5 = 0;
    sub_9C66B0(&v5);
  }
  return a1;
}
