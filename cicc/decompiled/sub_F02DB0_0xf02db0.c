// Function: sub_F02DB0
// Address: 0xf02db0
//
unsigned __int64 __fastcall sub_F02DB0(_DWORD *a1, unsigned int a2, unsigned int a3)
{
  unsigned __int64 result; // rax

  result = a2;
  if ( a3 != 0x80000000 )
    result = ((a3 >> 1) + ((unsigned __int64)a2 << 31)) / a3;
  *a1 = result;
  return result;
}
