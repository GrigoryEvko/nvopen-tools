// Function: sub_133DEF0
// Address: 0x133def0
//
__int64 __fastcall sub_133DEF0(__int64 a1, __int64 a2, unsigned int a3, int a4, char a5)
{
  unsigned int v8; // r12d

  v8 = sub_130AF40(a2);
  if ( !(_BYTE)v8 )
  {
    *(_DWORD *)(a2 + 19424) = a3;
    *(_DWORD *)(a2 + 19428) = a4;
    *(_BYTE *)(a2 + 19432) = a5;
    sub_1342750(a2 + 112, a3);
    sub_1342750(a2 + 9768, a3);
  }
  return v8;
}
