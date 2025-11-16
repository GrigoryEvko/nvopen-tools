// Function: sub_2CAFF80
// Address: 0x2caff80
//
bool __fastcall sub_2CAFF80(unsigned __int64 a1, __int64 a2, unsigned int a3, int a4)
{
  __int64 v4; // rax
  int v7; // r15d
  __int64 v8; // rcx
  unsigned int v9; // r8d
  unsigned int v10; // r15d
  bool result; // al
  int v12; // [rsp-3Ch] [rbp-3Ch] BYREF

  v4 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v4 + 8) != 12 )
    return 0;
  v7 = *(_DWORD *)(v4 + 8) >> 8;
  v10 = v7 - sub_9AF8B0(a2, a1, 0, 0, 0, 0, 1);
  result = 1;
  if ( v10 >= a3 )
  {
    if ( v10 > a3 )
    {
      return 0;
    }
    else
    {
      result = a4 == 2;
      if ( !a4 )
      {
        v12 = 1;
        return (unsigned __int8)sub_2CAFEF0(a1, (unsigned __int8 *)a2, &v12, v8, v9) == 0;
      }
    }
  }
  return result;
}
