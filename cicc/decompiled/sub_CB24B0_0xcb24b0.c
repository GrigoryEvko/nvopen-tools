// Function: sub_CB24B0
// Address: 0xcb24b0
//
__int64 __fastcall sub_CB24B0(__int64 a1, char *a2, char a3, char a4, _BYTE *a5, _QWORD *a6)
{
  __int64 result; // rax
  size_t v7; // rdx
  size_t v8; // rdx

  *a5 = 0;
  *a6 = 0;
  if ( a4 != 1 || a3 || (result = *(unsigned __int8 *)(a1 + 95), (_BYTE)result) )
  {
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * *(unsigned int *)(a1 + 40) - 4) - 6) > 1 )
    {
      sub_CB20A0(a1, 0);
      v8 = 0;
      if ( a2 )
        v8 = strlen(a2);
      sub_CB1E90(a1, a2, v8);
      return 1;
    }
    else
    {
      v7 = 0;
      if ( a2 )
        v7 = strlen(a2);
      sub_CB1F40(a1, a2, v7);
      return 1;
    }
  }
  return result;
}
