// Function: sub_7E4C20
// Address: 0x7e4c20
//
_QWORD *__fastcall sub_7E4C20(__int64 a1, __int64 a2, __int64 *a3)
{
  _QWORD *result; // rax
  __int64 v4; // rbx
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  _QWORD *v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  result = *(_QWORD **)(a1 + 152);
  v16 = result;
  if ( result && (*((_BYTE *)result + 29) & 0x20) == 0 )
  {
    v4 = result[12];
    while ( v4 )
    {
      v7 = v4;
      v4 = *(_QWORD *)(v4 + 120);
      sub_733310(v7, 1);
    }
    v8 = v16[14];
    v16[12] = 0;
    while ( v8 )
    {
      while ( 1 )
      {
        v9 = v8;
        v8 = *(_QWORD *)(v8 + 112);
        sub_735E40(v9, 0);
        if ( (*(_BYTE *)(v9 + 170) & 0x10) != 0 && !*(_BYTE *)(v9 + 136) )
          break;
        if ( !v8 )
          goto LABEL_11;
      }
      sub_7E4C10(v9);
    }
LABEL_11:
    v16[14] = 0;
    sub_7DFE30((__int64)v16);
    v17 = v16[13];
    if ( !v17 )
      goto LABEL_24;
LABEL_12:
    v10 = v17;
    v11 = 0;
    v17 = 0;
    while ( 1 )
    {
      v12 = v10;
      v10 = *(_QWORD *)(v10 + 112);
      if ( (unsigned __int8)(*(_BYTE *)(v12 + 140) - 9) > 2u || (unsigned int)sub_736DD0(v12) )
      {
        v13 = *a3;
        if ( *a3 )
          goto LABEL_16;
      }
      else
      {
        sub_7E4C20(*(_QWORD *)(v12 + 168), a2, a3);
        v13 = *a3;
        if ( *a3 )
        {
LABEL_16:
          *(_QWORD *)(v12 + 112) = *(_QWORD *)(v13 + 112);
          *(_QWORD *)(*a3 + 112) = v12;
          goto LABEL_17;
        }
      }
      *(_QWORD *)(v12 + 112) = *(_QWORD *)(a2 + 104);
      *(_QWORD *)(a2 + 104) = v12;
LABEL_17:
      *a3 = v12;
      if ( (unsigned __int8)(*(_BYTE *)(v12 + 140) - 9) <= 2u
        && (v14 = *(_QWORD *)(v12 + 168), (v15 = *(_QWORD *)(v14 + 216)) != 0) )
      {
        if ( v17 )
          *(_QWORD *)(v11 + 112) = v15;
        else
          v17 = *(_QWORD *)(v14 + 216);
        do
        {
          v11 = v15;
          v15 = *(_QWORD *)(v15 + 112);
        }
        while ( v15 );
        *(_QWORD *)(v14 + 216) = 0;
        if ( !v10 )
        {
LABEL_23:
          if ( !v17 )
          {
LABEL_24:
            v16[13] = 0;
            return sub_7DF680((__int64)v16);
          }
          goto LABEL_12;
        }
      }
      else if ( !v10 )
      {
        goto LABEL_23;
      }
    }
  }
  return result;
}
