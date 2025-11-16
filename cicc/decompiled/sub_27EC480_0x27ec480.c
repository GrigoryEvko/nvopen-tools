// Function: sub_27EC480
// Address: 0x27ec480
//
__int64 __fastcall sub_27EC480(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  int v9; // eax
  int v10; // esi
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rsi
  int v17; // eax

  v8 = *(_QWORD *)(*a3 + 40LL);
  v9 = *(_DWORD *)(*a3 + 56LL);
  if ( v9 )
  {
    v10 = v9 - 1;
    v12 = (v9 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v13 = (__int64 *)(v8 + 16LL * v12);
    v14 = *v13;
    if ( a1 == (_QWORD *)*v13 )
    {
LABEL_3:
      v15 = v13[1];
      if ( v15 )
        sub_D6E4B0(a3, v15, 0, v8, v14, a6);
    }
    else
    {
      v17 = 1;
      while ( v14 != -4096 )
      {
        a6 = (unsigned int)(v17 + 1);
        v12 = v10 & (v17 + v12);
        v13 = (__int64 *)(v8 + 16LL * v12);
        v14 = *v13;
        if ( a1 == (_QWORD *)*v13 )
          goto LABEL_3;
        v17 = a6;
      }
    }
  }
  sub_31032E0(a2);
  return sub_B43D60(a1);
}
