// Function: sub_2244ED0
// Address: 0x2244ed0
//
__int64 __fastcall sub_2244ED0(
        __int64 a1,
        char *a2,
        __int64 a3,
        int a4,
        const wchar_t *a5,
        _DWORD *a6,
        __int64 a7,
        int *a8)
{
  __int64 v10; // r13
  wchar_t *v11; // rdi
  __int64 v12; // r12
  int v13; // eax
  __int64 result; // rax

  if ( a5 )
  {
    v10 = ((__int64)a5 - a7) >> 2;
    v11 = sub_2244D30(a6, a4, a2, a3, a7, a7 + 4LL * (int)v10);
    v12 = v11 - a6;
    v13 = 0;
    if ( *a8 != (_DWORD)v10 )
    {
      wmemcpy(v11, a5, *a8 - (int)v10);
      v13 = *a8 - v10;
    }
    result = (unsigned int)(v12 + v13);
    *a8 = result;
  }
  else
  {
    result = sub_2244D30(a6, a4, a2, a3, a7, a7 + 4LL * *a8) - a6;
    *a8 = result;
  }
  return result;
}
