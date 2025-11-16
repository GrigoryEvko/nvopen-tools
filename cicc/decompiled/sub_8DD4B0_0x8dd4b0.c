// Function: sub_8DD4B0
// Address: 0x8dd4b0
//
__int64 __fastcall sub_8DD4B0(__int64 a1, int a2, __m128i *a3, __int64 a4, _DWORD *a5)
{
  __int64 v6; // r13
  __int64 v7; // r12
  bool v9; // zf
  unsigned int v10; // r14d
  char v12; // dl
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rax
  int v16[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = a4;
  v7 = a1;
  v9 = *(_BYTE *)(a1 + 140) == 12;
  v16[0] = 0;
  if ( !v9 )
    goto LABEL_5;
  do
    v7 = *(_QWORD *)(v7 + 160);
  while ( *(_BYTE *)(v7 + 140) == 12 );
  if ( *(_BYTE *)(a4 + 140) == 12 )
  {
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
LABEL_5:
      ;
    }
    while ( *(_BYTE *)(v6 + 140) == 12 );
  }
  if ( v6 == v7 )
    goto LABEL_8;
  v10 = sub_8D97D0(v7, v6, 0, a4, (__int64)a5);
  if ( v10 )
    goto LABEL_8;
  if ( word_4D04898 )
  {
    if ( (unsigned int)sub_8DD3B0(v7) || (unsigned int)sub_8DD3B0(v6) )
      goto LABEL_8;
  }
  else if ( sub_8D3D40(v7) || sub_8D3D40(v6) )
  {
    goto LABEL_8;
  }
  v12 = *(_BYTE *)(v7 + 140);
  if ( v12 == 12 )
  {
    v13 = v7;
    do
    {
      v13 = *(_QWORD *)(v13 + 160);
      v12 = *(_BYTE *)(v13 + 140);
    }
    while ( v12 == 12 );
  }
  if ( !v12 )
    goto LABEL_8;
  v14 = *(_BYTE *)(v6 + 140);
  if ( v14 == 12 )
  {
    v15 = v6;
    do
    {
      v15 = *(_QWORD *)(v15 + 160);
      v14 = *(_BYTE *)(v15 + 140);
    }
    while ( v14 == 12 );
  }
  if ( !v14 )
    goto LABEL_8;
  if ( !(unsigned int)sub_8D2960(v7) || !(unsigned int)sub_8D2780(v6) )
    goto LABEL_9;
  if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0xEA5Fu )
  {
LABEL_8:
    v10 = 1;
    goto LABEL_9;
  }
  if ( !a2 )
    a3 = 0;
  v10 = sub_8D67E0(v7, a3, v6, 0, v16) == 0;
LABEL_9:
  if ( a5 )
    *a5 = v16[0];
  return v10;
}
