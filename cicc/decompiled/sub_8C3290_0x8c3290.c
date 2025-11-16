// Function: sub_8C3290
// Address: 0x8c3290
//
__int64 __fastcall sub_8C3290(_QWORD *src, __int64 a2, size_t a3)
{
  __int64 result; // rax
  void *v4; // rax

  result = *((unsigned __int8 *)src - 8);
  if ( (result & 2) != 0 )
  {
    if ( (result & 1) != 0 )
    {
      v4 = (void *)sub_7279A0(a3);
      *(src - 3) = v4;
      return (__int64)memcpy(v4, src, a3);
    }
    else
    {
      result = (unsigned int)result & 0xFFFFFFFD;
      *((_BYTE *)src - 8) = result;
    }
  }
  return result;
}
