// Function: sub_8241D0
// Address: 0x8241d0
//
__int64 __fastcall sub_8241D0(FILE *stream)
{
  size_t v1; // r8
  __int64 result; // rax
  _BYTE ptr[12]; // [rsp+Ch] [rbp-14h] BYREF

  if ( fseek(stream, 0, 0) )
    sub_685240(0xC4Du);
  v1 = fread(ptr, 1u, 4u, stream);
  result = 0;
  if ( v1 == 4 && ptr[0] == 0x9A && ptr[1] == 19 && ptr[2] == 55 )
  {
    LOBYTE(result) = ptr[3] == 125;
    return (unsigned int)(2 * result);
  }
  return result;
}
