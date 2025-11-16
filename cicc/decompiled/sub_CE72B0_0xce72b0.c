// Function: sub_CE72B0
// Address: 0xce72b0
//
__int64 __fastcall sub_CE72B0(__int64 a1, int a2, char a3, int a4, char a5, int a6, char a7)
{
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // [rsp+0h] [rbp-Ch]

  result = a1;
  v9 = a1 + 16;
  if ( a3 )
  {
    LODWORD(v10) = a2;
    if ( !a5 )
      a4 = 1;
    goto LABEL_6;
  }
  if ( a5 )
  {
    LODWORD(v10) = 1;
LABEL_6:
    HIDWORD(v10) = a4;
    if ( !a7 )
      a6 = 1;
    goto LABEL_8;
  }
  if ( a7 )
  {
    v10 = 0x100000001LL;
LABEL_8:
    *(_QWORD *)a1 = v9;
    *(_QWORD *)(a1 + 16) = v10;
    *(_DWORD *)(a1 + 24) = a6;
    *(_QWORD *)(a1 + 8) = 0x300000003LL;
    return result;
  }
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 8) = 0x300000000LL;
  return result;
}
