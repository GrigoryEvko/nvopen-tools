// Function: sub_274B110
// Address: 0x274b110
//
__int64 __fastcall sub_274B110(_QWORD *a1, __int64 *a2)
{
  unsigned int v2; // r15d
  _BYTE *v5; // rdi
  __int64 v6; // rbx
  unsigned __int64 v7; // rax
  unsigned int v8; // r15d
  bool v9; // al
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r14

  if ( (unsigned int)*(unsigned __int8 *)(a1[1] + 8LL) - 17 <= 1 )
    return 0;
  v5 = (_BYTE *)*(a1 - 12);
  if ( *v5 <= 0x15u )
    return 0;
  v6 = a1[2];
  v2 = 0;
  if ( !v6 )
    return 0;
  while ( 1 )
  {
    v13 = *(_QWORD *)(v6 + 24);
    v14 = *(_QWORD *)(v6 + 8);
    if ( *(_BYTE *)v13 == 84 )
      v7 = (unsigned __int64)sub_22CF3A0(
                               a2,
                               (__int64)v5,
                               *(_QWORD *)(*(_QWORD *)(v13 - 8)
                                         + 32LL * *(unsigned int *)(v13 + 72)
                                         + 8LL * (unsigned int)((v6 - *(_QWORD *)(v13 - 8)) >> 5)),
                               *(_QWORD *)(v13 + 40),
                               v13);
    else
      v7 = sub_274B0A0((__int64)v5, *(_QWORD *)(v6 + 24), a2);
    if ( v7 && *(_BYTE *)v7 == 17 )
    {
      v8 = *(_DWORD *)(v7 + 32);
      if ( v8 <= 0x40 )
        v9 = *(_QWORD *)(v7 + 24) == 1;
      else
        v9 = v8 - 1 == (unsigned int)sub_C444A0(v7 + 24);
      if ( v9 )
        v10 = *(a1 - 8);
      else
        v10 = *(a1 - 4);
      if ( *(_QWORD *)v6 )
      {
        v11 = *(_QWORD *)(v6 + 8);
        **(_QWORD **)(v6 + 16) = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v6 + 16);
      }
      *(_QWORD *)v6 = v10;
      v2 = 1;
      if ( v10 )
      {
        v12 = *(_QWORD *)(v10 + 16);
        *(_QWORD *)(v6 + 8) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = v6 + 8;
        *(_QWORD *)(v6 + 16) = v10 + 16;
        v2 = 1;
        *(_QWORD *)(v10 + 16) = v6;
      }
    }
    if ( !v14 )
      break;
    v5 = (_BYTE *)*(a1 - 12);
    v6 = v14;
  }
  if ( (_BYTE)v2 )
  {
    if ( !a1[2] )
      sub_B43D60(a1);
  }
  else
  {
    return 0;
  }
  return v2;
}
