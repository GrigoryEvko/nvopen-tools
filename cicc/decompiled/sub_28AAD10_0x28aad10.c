// Function: sub_28AAD10
// Address: 0x28aad10
//
__int64 __fastcall sub_28AAD10(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  int v10; // eax
  int v11; // esi
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rsi
  int v17; // eax

  v8 = *(_QWORD **)(a1 + 48);
  v9 = *(_QWORD *)(*v8 + 40LL);
  v10 = *(_DWORD *)(*v8 + 56LL);
  if ( v10 )
  {
    v11 = v10 - 1;
    v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( a2 == (_QWORD *)*v13 )
    {
LABEL_3:
      v15 = v13[1];
      if ( v15 )
        sub_D6E4B0(v8, v15, 0, v9, v14, a6);
    }
    else
    {
      v17 = 1;
      while ( v14 != -4096 )
      {
        a6 = (unsigned int)(v17 + 1);
        v12 = v11 & (v17 + v12);
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( a2 == (_QWORD *)*v13 )
          goto LABEL_3;
        v17 = a6;
      }
    }
  }
  sub_D03960(*(_QWORD *)(a1 + 56), (__int64)a2);
  return sub_B43D60(a2);
}
