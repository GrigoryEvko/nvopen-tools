// Function: sub_34A3C90
// Address: 0x34a3c90
//
unsigned __int64 __fastcall sub_34A3C90(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned int v7; // esi
  unsigned __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 result; // rax
  __int64 v13; // rcx

  v7 = *(_DWORD *)(*(_QWORD *)a1 + 196LL);
  if ( v7 )
  {
    v8 = (unsigned __int64 *)(*(_QWORD *)a1 + 96LL);
    a3 = 0;
    while ( a2 > *v8 )
    {
      a3 = (unsigned int)(a3 + 1);
      ++v8;
      if ( v7 == (_DWORD)a3 )
        goto LABEL_6;
    }
    v7 = a3;
  }
LABEL_6:
  sub_34A26E0(a1, v7, a3, a4, a5, a6);
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 8);
    v13 = *(unsigned int *)(result + 8);
    if ( *(_DWORD *)(result + 12) < (unsigned int)v13 )
      return sub_34A3B20(a1, a2, v9, v13, v10, v11);
  }
  return result;
}
