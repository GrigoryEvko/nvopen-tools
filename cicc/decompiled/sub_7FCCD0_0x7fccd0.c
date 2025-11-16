// Function: sub_7FCCD0
// Address: 0x7fccd0
//
__int64 *__fastcall sub_7FCCD0(__int64 a1, __int64 a2, __int64 a3, __int64 **a4)
{
  __int64 *result; // rax

  result = *(__int64 **)(a1 + 152);
  *a4 = 0;
  if ( *(_BYTE *)(a1 + 129) )
  {
    result = sub_7FCC60(*(_QWORD *)(a1 + 144), *(unsigned __int8 *)(a1 + 130), *((_WORD *)result + 69));
    *a4 = result;
  }
  return result;
}
