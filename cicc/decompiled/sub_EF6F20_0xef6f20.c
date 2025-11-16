// Function: sub_EF6F20
// Address: 0xef6f20
//
__int64 __fastcall sub_EF6F20(unsigned __int8 ****a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 **v7; // rax
  unsigned __int8 *v8; // rcx
  unsigned __int8 *v9; // rcx
  unsigned __int8 *v10; // rcx
  __int64 v11; // rcx
  __int64 result; // rax
  __int64 v13; // rdi
  _BYTE *v14; // r12
  unsigned __int8 ***v15; // rax

  v7 = **a1;
  v7[1] = (unsigned __int8 *)(a2 + a3);
  v8 = v7[2];
  *v7 = (unsigned __int8 *)a2;
  v7[3] = v8;
  v9 = v7[37];
  v7[98] = (unsigned __int8 *)-1LL;
  v7[38] = v9;
  v10 = v7[83];
  v7[99] = 0;
  v7[84] = v10;
  *((_WORD *)v7 + 388) = 1;
  *((_DWORD *)v7 + 200) = 0;
  v7[115] = 0;
  v11 = *(unsigned int *)a1[1];
  if ( (_DWORD)v11 == 1 )
  {
    v13 = (__int64)**a1;
    goto LABEL_8;
  }
  if ( (_DWORD)v11 == 2 )
  {
    result = (__int64)sub_EF05F0(**a1, 1);
  }
  else
  {
    result = 0;
    if ( (_DWORD)v11 )
      goto LABEL_4;
    v14 = (_BYTE *)a2;
    if ( a3 != 2 )
    {
      if ( !a3 )
      {
        v15 = *a1;
LABEL_12:
        result = sub_EF1680((__int64)*v15, 0, a3, v11, a5, a6);
        goto LABEL_4;
      }
      goto LABEL_15;
    }
    a2 = 2;
    if ( !(unsigned __int8)sub_EE3B50((const void **)**a1, 2u, "St") )
    {
LABEL_15:
      v15 = *a1;
      v13 = (__int64)**a1;
      if ( *v14 != 83 )
        goto LABEL_12;
LABEL_8:
      result = sub_EF1F20(v13, a2, a3, v11, a5, a6);
      goto LABEL_4;
    }
    result = sub_EE68C0((__int64)(**a1 + 101), "std");
  }
LABEL_4:
  if ( (**a1)[1] != ***a1 )
    return 0;
  return result;
}
