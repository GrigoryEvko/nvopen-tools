// Function: sub_9C4420
// Address: 0x9c4420
//
__int64 __fastcall sub_9C4420(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        int a4,
        __int64 a5,
        unsigned int a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v15; // rax
  _QWORD *v16; // [rsp+8h] [rbp-18h]

  v10 = (unsigned int)*a3;
  if ( (_DWORD)v10 == *(_DWORD *)(a2 + 8) )
  {
    *a7 = 0;
    return 1;
  }
  else
  {
    v11 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v10);
    v12 = (unsigned int)v11;
    if ( *(_BYTE *)(a1 + 1832) )
      v12 = (unsigned int)(a4 - v11);
    if ( a5 && *(_BYTE *)(a5 + 8) == 9 )
    {
      v16 = (_QWORD *)a5;
      v15 = sub_A12C40(a1 + 808, v12, a5, a6, v10);
      v13 = sub_B9F6F0(*v16, v15);
    }
    else
    {
      v13 = sub_A14C90(a1 + 744, v12, a5, a6, a8);
    }
    *a7 = v13;
    if ( v13 )
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
