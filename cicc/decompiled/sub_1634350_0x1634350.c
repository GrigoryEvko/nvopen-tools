// Function: sub_1634350
// Address: 0x1634350
//
__int64 __fastcall sub_1634350(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r9
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned int v8; // r8d
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx

  v2 = *(_QWORD **)(a1 + 16);
  v3 = (_QWORD *)(a1 + 8);
  v5 = (_QWORD *)(a1 + 8);
  if ( v2 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v2[2];
        v7 = v2[3];
        if ( a2 <= v2[4] )
          break;
        v2 = (_QWORD *)v2[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v2;
      v2 = (_QWORD *)v2[2];
    }
    while ( v6 );
LABEL_6:
    if ( v3 != v5 && a2 >= v5[4] )
    {
      v10 = (unsigned __int16)(4 * *(unsigned __int8 *)(a1 + 178)) & 0xFFF8
          | (unsigned __int64)(v5 + 4) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v10 )
      {
        v11 = *(_QWORD *)(v10 + 24);
        v12 = *(_QWORD *)(v10 + 32);
        if ( v11 != v12 )
        {
          v8 = *(unsigned __int8 *)(a1 + 176);
          while ( (_BYTE)v8 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)v11 + 12LL) & 0x20) != 0 )
              return v8;
            v11 += 8;
            if ( v12 == v11 )
              return 0;
          }
        }
      }
    }
  }
  return 1;
}
