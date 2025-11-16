// Function: sub_120EC00
// Address: 0x120ec00
//
__int64 __fastcall sub_120EC00(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 v4; // rdi
  unsigned __int64 v5; // rsi
  unsigned int v8; // r14d
  const char *v9; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  v4 = a1 + 176;
  if ( (_BYTE)a3 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(v4);
    if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
      return a3;
    v8 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v8 )
    {
      return a3;
    }
    else
    {
      *a2 = sub_AF40E0(*(_QWORD *)a1, 1u);
      return v8;
    }
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 232);
    v9 = "missing 'distinct', required for !DIAssignID()";
    v11 = 1;
    v10 = 3;
    sub_11FD800(v4, v5, (__int64)&v9, 1);
    return 1;
  }
}
