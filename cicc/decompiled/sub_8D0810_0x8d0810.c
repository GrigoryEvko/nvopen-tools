// Function: sub_8D0810
// Address: 0x8d0810
//
__int64 __fastcall sub_8D0810(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 16);
  if ( v1 == 1 )
  {
    result = qword_4F60548;
    qword_4F60548 = a1;
    *(_QWORD *)a1 = result;
  }
  else
  {
    result = (unsigned int)(v1 - 1);
    *(_DWORD *)(a1 + 16) = result;
  }
  return result;
}
