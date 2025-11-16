// Function: sub_7F54F0
// Address: 0x7f54f0
//
__int64 *__fastcall sub_7F54F0(__int64 a1, int a2, int a3, _DWORD *a4)
{
  int v6; // r14d
  int v7; // eax
  __int64 *v8; // r12
  __int64 i; // rax
  _BYTE *v10; // rax

  v6 = dword_4F07270[0];
  v7 = sub_880E90();
  v8 = sub_729790(v7, a1, a3);
  v8[2] = qword_4F07288;
  *a4 = dword_4F07270[0];
  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  *(_QWORD *)(*(_QWORD *)(i + 168) + 8LL) = a1;
  if ( *(_BYTE *)(a1 + 172) == 1 )
    *(_BYTE *)(a1 + 172) = 0;
  v10 = sub_726B30(11);
  v8[10] = (__int64)v10;
  *(_BYTE *)(*((_QWORD *)v10 + 10) + 24LL) &= ~1u;
  if ( a2 )
    *((_QWORD *)v10 + 9) = sub_726B30(8);
  sub_7296B0(v6);
  return v8;
}
