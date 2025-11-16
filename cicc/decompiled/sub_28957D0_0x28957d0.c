// Function: sub_28957D0
// Address: 0x28957d0
//
__int64 __fastcall sub_28957D0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v4; // rsi
  unsigned int v5; // edx
  _QWORD *v6; // rcx
  _QWORD *v7; // r9
  int v9; // ecx
  int v10; // r11d

  v2 = *(unsigned int *)(a1 + 88);
  v4 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v2 )
  {
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (_QWORD *)(v4 + 24LL * v5);
    v7 = (_QWORD *)*v6;
    if ( a2 == (_QWORD *)*v6 )
    {
LABEL_3:
      if ( v6 != (_QWORD *)(v4 + 24 * v2) )
      {
        *v6 = -8192;
        --*(_DWORD *)(a1 + 80);
        ++*(_DWORD *)(a1 + 84);
      }
    }
    else
    {
      v9 = 1;
      while ( v7 != (_QWORD *)-4096LL )
      {
        v10 = v9 + 1;
        v5 = (v2 - 1) & (v9 + v5);
        v6 = (_QWORD *)(v4 + 24LL * v5);
        v7 = (_QWORD *)*v6;
        if ( a2 == (_QWORD *)*v6 )
          goto LABEL_3;
        v9 = v10;
      }
    }
  }
  return sub_B43D60(a2);
}
