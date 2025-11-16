// Function: sub_250EEA0
// Address: 0x250eea0
//
__int64 __fastcall sub_250EEA0(__int64 a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 result; // rax

  v2 = sub_C52410();
  v3 = v2 + 1;
  v4 = sub_C959E0();
  v5 = (_QWORD *)v2[2];
  if ( v5 )
  {
    v6 = v2 + 1;
    do
    {
      while ( 1 )
      {
        v7 = v5[2];
        v8 = v5[3];
        if ( v4 <= v5[4] )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_6:
    if ( v3 != v6 && v4 >= v6[4] )
      v3 = v6;
  }
  if ( v3 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_19;
  v9 = v3[7];
  if ( !v9 )
    goto LABEL_19;
  v10 = v3 + 6;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v9 + 16);
      v12 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= dword_4FEE4E8 )
        break;
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v12 )
        goto LABEL_15;
    }
    v10 = (_QWORD *)v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v11 );
LABEL_15:
  if ( v3 + 6 == v10
    || dword_4FEE4E8 < *((_DWORD *)v10 + 8)
    || (result = (unsigned __int8)qword_4FEE568, !*((_DWORD *)v10 + 9)) )
  {
LABEL_19:
    result = *(unsigned __int8 *)(a1 + 4296);
    if ( (_BYTE)result )
      return *(unsigned __int8 *)(a1 + 4301);
  }
  return result;
}
