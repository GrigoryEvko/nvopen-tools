// Function: sub_16E4D00
// Address: 0x16e4d00
//
__int64 __fastcall sub_16E4D00(__int64 a1, const char *a2, size_t a3)
{
  __int64 result; // rax

  sub_16E4B40(a1, a2, a3);
  result = *(unsigned int *)(a1 + 40);
  if ( !(_DWORD)result
    || (result = *(unsigned int *)(*(_QWORD *)(a1 + 32) + 4 * result - 4), (unsigned int)(result - 4) > 1)
    && (_DWORD)result != 1 )
  {
    *(_BYTE *)(a1 + 95) = 1;
  }
  return result;
}
