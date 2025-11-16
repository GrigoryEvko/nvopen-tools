// Function: sub_2895E10
// Address: 0x2895e10
//
__int64 __fastcall sub_2895E10(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // edx
  _QWORD *v10; // rcx
  _QWORD *v11; // r9
  int v13; // ecx
  int v14; // r11d

  v4 = *(_QWORD **)a3;
  if ( *(_QWORD *)a3 != a4 + 48 )
  {
    v6 = v4 - 3;
    if ( !v4 )
      v6 = 0;
    if ( a2 == v6 )
    {
      *(_QWORD *)a3 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
      *(_WORD *)(a3 + 8) = 0;
    }
  }
  v7 = *(unsigned int *)(a1 + 88);
  v8 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (_QWORD *)(v8 + 24LL * v9);
    v11 = (_QWORD *)*v10;
    if ( a2 == (_QWORD *)*v10 )
    {
LABEL_8:
      if ( v10 != (_QWORD *)(v8 + 24 * v7) )
      {
        *v10 = -8192;
        --*(_DWORD *)(a1 + 80);
        ++*(_DWORD *)(a1 + 84);
      }
    }
    else
    {
      v13 = 1;
      while ( v11 != (_QWORD *)-4096LL )
      {
        v14 = v13 + 1;
        v9 = (v7 - 1) & (v13 + v9);
        v10 = (_QWORD *)(v8 + 24LL * v9);
        v11 = (_QWORD *)*v10;
        if ( a2 == (_QWORD *)*v10 )
          goto LABEL_8;
        v13 = v14;
      }
    }
  }
  return sub_B43D60(a2);
}
