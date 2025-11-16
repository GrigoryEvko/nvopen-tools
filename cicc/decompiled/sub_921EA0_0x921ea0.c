// Function: sub_921EA0
// Address: 0x921ea0
//
__int64 __fastcall sub_921EA0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned int a5, unsigned __int8 a6)
{
  __int64 v10; // rax
  char v12; // al

  if ( sub_91B770(*a3) )
  {
    sub_947E80(a2, a3, a4, a5, a6);
    v12 = *(_BYTE *)(a1 + 12);
    *(_QWORD *)a1 = a4;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 16) = a5;
    *(_BYTE *)(a1 + 12) = v12 & 0xFE | a6 & 1;
  }
  else
  {
    v10 = sub_92F410(a2, a3);
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = v10;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
  }
  return a1;
}
