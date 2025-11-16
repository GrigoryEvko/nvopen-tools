// Function: sub_771840
// Address: 0x771840
//
__int64 __fastcall sub_771840(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r10
  __int64 v10; // r11
  __int64 v12; // r9
  __int64 v13; // rdi
  _QWORD *v14; // rsi
  __int64 v15; // rax
  __int64 i; // rax

  result = 1;
  v7 = *(unsigned int *)(a3 + 4);
  if ( (_DWORD)v7 )
  {
    v8 = *a2;
    if ( (*(_BYTE *)a3 & 2) == 0 )
    {
      *a2 = (__int64 *)((char *)v8 + v7);
      return result;
    }
    v10 = *v8;
    v12 = **(_QWORD **)(a3 + 8);
    if ( *v8 )
    {
      v13 = *(_QWORD *)(v12 + 64);
      a4 = *(_QWORD *)(v10 + 56);
      if ( v13 == a4 )
        goto LABEL_13;
      v14 = *(_QWORD **)(*(_QWORD *)(v10 + 112) + 8LL);
      v15 = v14[2];
      if ( v15 != v10 )
      {
        while ( 1 )
        {
          for ( i = *(_QWORD *)(v15 + 40); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          if ( v13 == i )
            break;
          v14 = (_QWORD *)*v14;
          v15 = v14[2];
          if ( v15 == v10 )
            goto LABEL_10;
        }
        result = 1;
        goto LABEL_13;
      }
LABEL_10:
      result = 1;
      if ( !a4 )
      {
LABEL_13:
        *a2 = (__int64 *)((char *)v8 - (unsigned int)v7);
        return result;
      }
    }
    else if ( *(_BYTE *)(a4 + 140) == 6 )
    {
      do
      {
        a4 = *(_QWORD *)(a4 + 160);
        if ( *(_BYTE *)(a4 + 140) != 12 )
          break;
        a4 = *(_QWORD *)(a4 + 160);
      }
      while ( *(_BYTE *)(a4 + 140) == 12 );
    }
    result = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_687670(0xA99u, a5 + 28, v12, a4, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      v8 = *a2;
      LODWORD(v7) = *(_DWORD *)(a3 + 4);
      result = 0;
    }
    goto LABEL_13;
  }
  return result;
}
