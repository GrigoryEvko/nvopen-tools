// Function: sub_D79DB0
// Address: 0xd79db0
//
__int64 __fastcall sub_D79DB0(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4, unsigned int a5)
{
  int v5; // eax
  int v6; // r14d
  __int64 v7; // r9
  __int64 v8; // r11
  int v9; // edi
  __int64 v10; // r12
  unsigned int v11; // ebx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r10
  _QWORD *v15; // r13

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = 1;
  v7 = *a2;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v5 - 1;
  v10 = a2[1];
  v11 = *a2 & (v5 - 1);
  v12 = (_QWORD *)(v8 + 16LL * v11);
  v13 = *v12;
  v14 = v12[1];
  v15 = 0;
  LOBYTE(a5) = v10 == v14 && v7 == *v12;
  if ( (_BYTE)a5 )
  {
LABEL_8:
    *a3 = v12;
    return 1;
  }
  while ( 1 )
  {
    if ( v13 )
      goto LABEL_4;
    if ( v14 == -1 )
      break;
    if ( !v15 && v14 == -2 )
      v15 = v12;
LABEL_4:
    v11 = v9 & (v6 + v11);
    v12 = (_QWORD *)(v8 + 16LL * v11);
    v14 = v12[1];
    v13 = *v12;
    if ( v14 == v10 && v7 == v13 )
      goto LABEL_8;
    ++v6;
  }
  if ( !v15 )
    v15 = v12;
  *a3 = v15;
  return a5;
}
