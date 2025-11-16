// Function: sub_5E6230
// Address: 0x5e6230
//
__int64 __fastcall sub_5E6230(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, _DWORD *a5)
{
  __int64 v7; // r8
  __int64 result; // rax
  __int64 i; // rbx
  __int64 v11; // rsi
  __int64 v12; // rbx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // [rsp+0h] [rbp-40h]
  unsigned int v16; // [rsp+Ch] [rbp-34h]

  v7 = a1;
  if ( a2 )
    v7 = *(_QWORD *)(a2 + 40);
  result = *(_QWORD *)(v7 + 168);
  if ( *(char *)(v7 + 176) < 0 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(result + 152) + 144LL);
    if ( v12 )
    {
      do
      {
        if ( (*(_BYTE *)(v12 + 192) & 8) != 0 )
        {
          if ( !a2 )
            goto LABEL_22;
          v13 = *(_QWORD **)(a2 + 120);
          if ( !v13 )
            goto LABEL_22;
          while ( 1 )
          {
            v14 = v13[2];
            if ( v14 == v12 )
              break;
            if ( *(_WORD *)(v14 + 224) <= *(_WORD *)(v12 + 224) )
            {
              v13 = (_QWORD *)*v13;
              if ( v13 )
                continue;
            }
            goto LABEL_22;
          }
          if ( v13[1] == v12 )
          {
LABEL_22:
            v15 = v7;
            v16 = a4;
            sub_67E1D0(a3, a4, *(_QWORD *)v12);
            *a5 = 1;
            v7 = v15;
            a4 = v16;
          }
        }
        v12 = *(_QWORD *)(v12 + 112);
      }
      while ( v12 );
      result = *(_QWORD *)(v7 + 168);
    }
  }
  for ( i = *(_QWORD *)result; i; i = *(_QWORD *)i )
  {
    result = *(unsigned __int8 *)(i + 96);
    if ( (*(_BYTE *)(*(_QWORD *)(i + 40) + 176LL) & 0x20) != 0 && (result & 2) != 0 )
    {
      if ( !a2 )
      {
        v11 = i;
LABEL_12:
        result = sub_5E6230(a1, v11, a3, 859, a5);
      }
    }
    else if ( (result & 1) != 0 )
    {
      v11 = i;
      if ( a2 )
        v11 = sub_8E5310(i, a1, a2);
      goto LABEL_12;
    }
  }
  return result;
}
