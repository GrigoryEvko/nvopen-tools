// Function: sub_16E55E0
// Address: 0x16e55e0
//
__int64 __fastcall sub_16E55E0(__int64 a1, const char *a2, char a3, char a4, _BYTE *a5)
{
  __int64 result; // rax
  size_t v6; // rdx
  size_t v7; // rdx

  *a5 = 0;
  if ( a4 != 1 || a3 || (result = *(unsigned __int8 *)(a1 + 96), (_BYTE)result) )
  {
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * *(unsigned int *)(a1 + 40) - 4) - 4) > 1 )
    {
      sub_16E4E00(a1);
      v7 = 0;
      if ( a2 )
        v7 = strlen(a2);
      sub_16E5480(a1, a2, v7);
      return 1;
    }
    else
    {
      v6 = 0;
      if ( a2 )
        v6 = strlen(a2);
      sub_16E54F0(a1, a2, v6);
      return 1;
    }
  }
  return result;
}
