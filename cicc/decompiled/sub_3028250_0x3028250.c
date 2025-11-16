// Function: sub_3028250
// Address: 0x3028250
//
__int64 __fastcall sub_3028250(_BYTE **a1, __int64 a2, unsigned int a3, _BYTE *a4, __int64 a5)
{
  __int64 result; // rax
  _BYTE *v7; // rax
  _BYTE *v8; // rax
  unsigned int v9; // [rsp+Ch] [rbp-24h]

  if ( !a4 || (result = 1, !*a4) )
  {
    v7 = *(_BYTE **)(a5 + 32);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(a5 + 24) )
    {
      v9 = a3;
      sub_CB5D20(a5, 91);
      a3 = v9;
    }
    else
    {
      *(_QWORD *)(a5 + 32) = v7 + 1;
      *v7 = 91;
    }
    sub_3028140(a1, a2, a3, a5, 0);
    v8 = *(_BYTE **)(a5 + 32);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(a5 + 24) )
    {
      sub_CB5D20(a5, 93);
      return 0;
    }
    else
    {
      *(_QWORD *)(a5 + 32) = v8 + 1;
      *v8 = 93;
      return 0;
    }
  }
  return result;
}
