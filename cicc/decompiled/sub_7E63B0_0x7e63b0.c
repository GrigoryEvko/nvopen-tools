// Function: sub_7E63B0
// Address: 0x7e63b0
//
__int64 __fastcall sub_7E63B0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  result = *(_QWORD *)(a1 + 168);
  if ( !a2 )
  {
    v11 = a1;
    if ( (*(_BYTE *)(a1 + 176) & 0x50) == 0 )
      goto LABEL_4;
LABEL_11:
    sub_7E5FC0(a1, a2, 0, *(_QWORD *)(result + 192), a3, a4);
    result = *(_QWORD *)(v11 + 168);
    goto LABEL_4;
  }
  v11 = *(_QWORD *)(a2 + 40);
  if ( *(_QWORD *)(a2 + 128) != -1 )
    goto LABEL_11;
  result = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL);
LABEL_4:
  v7 = *(_QWORD *)(result + 16);
  if ( v7 )
  {
    do
    {
      result = *(_BYTE *)(v7 + 96) & 3;
      if ( (_BYTE)result == 1 )
      {
        v8 = sub_8E5310(v7, a1, a2);
        result = sub_7E63B0(a1, v8, a3, a4);
      }
      v7 = *(_QWORD *)(v7 + 16);
    }
    while ( v7 );
    if ( !a2 )
    {
      result = *(_QWORD *)(v11 + 168);
      v9 = *(_QWORD *)(result + 16);
      if ( v9 )
      {
        if ( (*(_BYTE *)(v9 + 96) & 2) != 0 )
          goto LABEL_16;
        while ( 1 )
        {
          v9 = *(_QWORD *)(v9 + 16);
          if ( !v9 )
            break;
          if ( (*(_BYTE *)(v9 + 96) & 2) != 0 )
          {
LABEL_16:
            v10 = sub_8E5310(v9, a1, 0);
            result = sub_7E63B0(a1, v10, a3, a4);
          }
        }
      }
    }
  }
  return result;
}
