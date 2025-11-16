// Function: sub_2F3BFF0
// Address: 0x2f3bff0
//
__int64 __fastcall sub_2F3BFF0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // r14
  _QWORD *v5; // r13
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rax

  v4 = sub_C52410();
  v5 = v4 + 1;
  v6 = sub_C959E0();
  v7 = (_QWORD *)v4[2];
  if ( v7 )
  {
    v8 = v4 + 1;
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v6 <= v7[4] )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_6:
    if ( v5 != v8 && v6 >= v8[4] )
      v5 = v8;
  }
  if ( v5 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_21;
  v11 = v5[7];
  if ( !v11 )
    goto LABEL_21;
  v12 = v5 + 6;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      v14 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= dword_5023368 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v14 )
        goto LABEL_15;
    }
    v12 = (_QWORD *)v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v13 );
LABEL_15:
  if ( v5 + 6 == v12 || dword_5023368 < *((_DWORD *)v12 + 8) || (v15 = qword_50233E8, !*((_DWORD *)v12 + 9)) )
LABEL_21:
    v15 = sub_DFDBE0(a2);
  v16 = *(_QWORD **)(a1 + 24);
  if ( *(_DWORD *)(a1 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  LOBYTE(v16) = v15 < (unsigned __int64)v16;
  LOBYTE(v15) = v15 == 0;
  return (unsigned int)v15 | (unsigned int)v16;
}
