// Function: sub_2DEB2E0
// Address: 0x2deb2e0
//
_QWORD *__fastcall sub_2DEB2E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *result; // rax
  _QWORD *v4; // rcx
  _QWORD *v5; // rsi
  _QWORD *v6; // rcx
  _QWORD *v7; // rdx

  *(_QWORD *)a1 = off_49D4228;
  *(_QWORD *)(a1 + 48) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 32;
  *(_QWORD *)(a1 + 96) = a1 + 80;
  *(_QWORD *)(a1 + 104) = a1 + 80;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  v2 = *(unsigned int *)(a2 + 32);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 136) = a2;
  result = (_QWORD *)sub_2207820(152 * v2 + 8);
  if ( result )
  {
    *result = v2;
    v4 = result;
    v5 = result + 1;
    if ( v2 )
    {
      ++result;
      v6 = &v4[19 * v2 + 1];
      do
      {
        v7 = result + 4;
        *(_DWORD *)result = -1;
        result += 19;
        *(result - 18) = 0;
        *(result - 17) = v7;
        *((_DWORD *)result - 32) = 0;
        *((_DWORD *)result - 31) = 4;
        *((_DWORD *)result - 4) = 1;
        *(result - 3) = 0;
        *(result - 1) = 0;
      }
      while ( result != v6 );
    }
  }
  else
  {
    v5 = 0;
  }
  *(_QWORD *)(a1 + 128) = v5;
  return result;
}
