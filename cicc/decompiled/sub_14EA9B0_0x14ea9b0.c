// Function: sub_14EA9B0
// Address: 0x14ea9b0
//
__int64 __fastcall sub_14EA9B0(__int64 a1, __int64 a2, _DWORD *a3, int a4, __int64 a5, __int64 *a6)
{
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v14; // rax
  _QWORD *v15; // [rsp+8h] [rbp-18h]

  v9 = (unsigned int)*a3;
  if ( (_DWORD)v9 == *(_DWORD *)(a2 + 8) )
  {
    *a6 = 0;
    return 1;
  }
  else
  {
    v10 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v9);
    v11 = (unsigned int)v10;
    if ( *(_BYTE *)(a1 + 1656) )
      v11 = (unsigned int)(a4 - v10);
    if ( a5 && *(_BYTE *)(a5 + 8) == 8 )
    {
      v15 = (_QWORD *)a5;
      v14 = sub_1521F50(a1 + 608, v11);
      v12 = sub_1628DA0(*v15, v14);
    }
    else
    {
      v12 = sub_1522F40(a1 + 552, v11);
    }
    *a6 = v12;
    if ( v12 )
    {
      ++*a3;
      return 0;
    }
    else
    {
      return 1;
    }
  }
}
