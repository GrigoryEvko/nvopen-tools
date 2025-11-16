// Function: sub_AF5A10
// Address: 0xaf5a10
//
const char *__fastcall sub_AF5A10(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int64 v2; // rcx
  __int64 v3; // rax
  const char *result; // rax
  __int64 v5; // rdi
  unsigned __int8 v6; // al
  __int64 v7; // rdi

  v2 = *a1;
  if ( (unsigned __int8)v2 <= 0x24u )
  {
    v3 = 0x140000F000LL;
    if ( _bittest64(&v3, v2) )
    {
      result = (const char *)sub_AF5140((__int64)a1, 2u);
      if ( !result )
        return result;
      return (const char *)sub_B91420(result, 2);
    }
  }
  if ( (_BYTE)v2 == 18 || (_BYTE)v2 == 21 )
  {
    result = (const char *)sub_AF5140((__int64)a1, 2u);
    if ( !result )
      return result;
    return (const char *)sub_B91420(result, 2);
  }
  if ( (_BYTE)v2 == 33 )
  {
    v5 = *((_QWORD *)sub_A17150(a1 - 16) + 2);
    if ( !v5 )
      return (const char *)v5;
    return (const char *)sub_B91420(v5, a2);
  }
  result = byte_3F871B3;
  if ( (_BYTE)v2 == 22 )
  {
    v6 = *(a1 - 16);
    if ( (v6 & 2) != 0 )
      v7 = *((_QWORD *)a1 - 4);
    else
      v7 = (__int64)&a1[-8 * ((v6 >> 2) & 0xF) - 16];
    v5 = *(_QWORD *)(v7 + 16);
    if ( !v5 )
      return (const char *)v5;
    return (const char *)sub_B91420(v5, a2);
  }
  return result;
}
