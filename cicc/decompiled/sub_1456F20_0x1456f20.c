// Function: sub_1456F20
// Address: 0x1456f20
//
__int64 __fastcall sub_1456F20(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 *v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // r15
  __int64 v8; // r12

  while ( *(_BYTE *)(sub_1456040(a2) + 8) == 15 )
  {
    while ( 1 )
    {
      v3 = *(unsigned __int16 *)(a2 + 24);
      if ( (unsigned __int16)(v3 - 1) > 2u )
        break;
      a2 = *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)(sub_1456040(a2) + 8) != 15 )
        return a2;
    }
    if ( (unsigned __int16)(v3 - 7) > 2u && (unsigned int)(v3 - 4) > 1 )
      break;
    v5 = *(__int64 **)(a2 + 32);
    v6 = &v5[*(_QWORD *)(a2 + 40)];
    if ( v5 == v6 )
      break;
    v7 = 0;
    do
    {
      while ( 1 )
      {
        v8 = *v5;
        if ( *(_BYTE *)(sub_1456040(*v5) + 8) == 15 )
          break;
        if ( v6 == ++v5 )
          goto LABEL_13;
      }
      if ( v7 )
        return a2;
      ++v5;
      v7 = v8;
    }
    while ( v6 != v5 );
LABEL_13:
    if ( !v7 )
      break;
    a2 = v7;
  }
  return a2;
}
