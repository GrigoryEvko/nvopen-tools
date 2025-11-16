// Function: sub_C926D0
// Address: 0xc926d0
//
__int64 __fastcall sub_C926D0(__int64 a1, int a2, int a3)
{
  unsigned __int64 v3; // rsi
  unsigned __int64 v4; // rsi
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 20) = a3;
  if ( a2 )
  {
    v3 = (4 * a2 / 3u + 1) | ((unsigned __int64)(4 * a2 / 3u + 1) >> 1);
    v4 = (((v3 >> 2) | v3) >> 4) | (v3 >> 2) | v3;
    return sub_C92620(a1, ((((v4 >> 8) | v4) >> 16) | (v4 >> 8) | v4) + 1);
  }
  return result;
}
