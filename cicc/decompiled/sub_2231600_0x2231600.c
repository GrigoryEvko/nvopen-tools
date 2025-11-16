// Function: sub_2231600
// Address: 0x2231600
//
__int64 __fastcall sub_2231600(
        __int64 a1,
        char *a2,
        __int64 a3,
        char a4,
        const void *a5,
        _BYTE *a6,
        __int64 a7,
        int *a8)
{
  int v9; // ebx
  _BYTE *v10; // rax
  int v11; // r14d
  void *v12; // rdi
  int v13; // eax
  int v14; // r14d
  __int64 result; // rax

  v9 = (int)a6;
  if ( a5 )
  {
    v10 = sub_2231480(a6, a4, a2, a3, a7, a7 + (int)a5 - (int)a7);
    v11 = (int)v10;
    v12 = v10;
    v13 = 0;
    v14 = v11 - v9;
    if ( *a8 != (_DWORD)a5 - (_DWORD)a7 )
    {
      memcpy(v12, a5, *a8 - ((int)a5 - (int)a7));
      v13 = *a8 - ((_DWORD)a5 - a7);
    }
    result = (unsigned int)(v14 + v13);
    *a8 = result;
  }
  else
  {
    result = (unsigned int)sub_2231480(a6, a4, a2, a3, a7, a7 + *a8) - (unsigned int)a6;
    *a8 = result;
  }
  return result;
}
