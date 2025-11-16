// Function: sub_2103DC0
// Address: 0x2103dc0
//
__int64 __fastcall sub_2103DC0(__int64 *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  int i; // ecx
  __int64 v5; // rdx
  unsigned int v6; // esi
  _WORD *v7; // rdx
  _WORD *v8; // rsi
  unsigned __int64 v9; // rcx
  _WORD *v10; // rdx
  _QWORD *v11; // rax

  result = sub_1E6A620(a2);
  if ( result )
  {
    v3 = result;
    for ( i = *(unsigned __int16 *)result; (_WORD)i; v3 += 2 )
    {
      v5 = *a1;
      if ( !*a1 )
        BUG();
      v6 = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 24LL * (unsigned __int16)i + 16);
      result = v6 & 0xF;
      v9 = (unsigned int)(result * i);
      v7 = (_WORD *)(*(_QWORD *)(v5 + 56) + 2LL * (v6 >> 4));
      v8 = v7 + 1;
      LOWORD(v9) = *v7 + v9;
LABEL_6:
      v10 = v8;
      while ( v10 )
      {
        ++v10;
        v11 = (_QWORD *)(a1[1] + ((v9 >> 3) & 0x1FF8));
        *v11 |= 1LL << v9;
        result = (unsigned __int16)*(v10 - 1);
        v8 = 0;
        v9 = (unsigned int)(result + v9);
        if ( !(_WORD)result )
          goto LABEL_6;
      }
      i = *(unsigned __int16 *)(v3 + 2);
    }
  }
  return result;
}
