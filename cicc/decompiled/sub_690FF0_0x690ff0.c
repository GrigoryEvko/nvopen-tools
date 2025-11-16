// Function: sub_690FF0
// Address: 0x690ff0
//
__int64 __fastcall sub_690FF0(__int64 a1, int a2, int a3, int a4, int a5, __int64 a6, int a7, __int64 a8, __int64 a9)
{
  unsigned __int64 v9; // rax
  __int64 v10; // r11
  int v11; // eax

  if ( !a1
    || (v9 = *(unsigned __int8 *)(a1 + 80), (unsigned __int8)v9 <= 0x14u) && (v10 = 1182720, _bittest64(&v10, v9))
    || *(_DWORD *)(a9 + 80) )
  {
    v11 = a7 | 0x102;
  }
  else
  {
    v11 = a7 | 2;
  }
  return sub_8A55D0(a1, a2, a3, 0, a4, a5, a6, v11, a8, a9);
}
