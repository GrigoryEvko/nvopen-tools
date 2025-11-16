// Function: sub_40E525
// Address: 0x40e525
//
__int64 __fastcall sub_40E525(unsigned int *a1, __int64 a2, int a3, int a4, int a5, int a6, char a7)
{
  __int64 result; // rax
  int v8; // [rsp-18h] [rbp-18h]

  result = *a1;
  if ( (unsigned int)result <= 1 )
  {
    v8 = a4;
    --a1[6];
    *((_BYTE *)a1 + 28) = 1;
    if ( (_DWORD)result != 1 )
    {
      sub_130F0B0((_DWORD)a1, (unsigned int)"\n", a3, a4, a5, a6, a4);
      sub_130F150(a1);
    }
    return sub_130F0B0((_DWORD)a1, (unsigned int)"]", v8, a4, a5, a6, a7);
  }
  return result;
}
