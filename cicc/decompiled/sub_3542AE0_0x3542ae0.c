// Function: sub_3542AE0
// Address: 0x3542ae0
//
__int64 __fastcall sub_3542AE0(_DWORD *a1)
{
  __int64 result; // rax

  result = (unsigned int)qword_503EC08;
  if ( (int)qword_503EC08 > 0 || (result = (unsigned int)a1[878], (_DWORD)result) )
  {
    a1[869] = result;
  }
  else
  {
    result = (unsigned int)(a1[868] + qword_503E428);
    a1[869] = result;
  }
  return result;
}
