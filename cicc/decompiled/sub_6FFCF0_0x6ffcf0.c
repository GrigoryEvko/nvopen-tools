// Function: sub_6FFCF0
// Address: 0x6ffcf0
//
void __fastcall sub_6FFCF0(_BYTE *a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  if ( !*a2 )
  {
    sub_6FF940(a1, (__int64)a2, a3, a4, a5, a6);
    *a2 = 1;
    a1[18] |= 2u;
  }
}
