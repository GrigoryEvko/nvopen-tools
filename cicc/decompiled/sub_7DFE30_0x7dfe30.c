// Function: sub_7DFE30
// Address: 0x7dfe30
//
__int64 __fastcall sub_7DFE30(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 result; // rax
  _QWORD *v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 144);
  while ( v2 )
  {
    while ( 1 )
    {
      v3 = v2;
      v2 = *(_QWORD *)(v2 + 112);
      sub_7362F0(v3, 0);
      result = (unsigned int)*(unsigned __int8 *)(v3 + 174) - 1;
      if ( (unsigned __int8)(*(_BYTE *)(v3 + 174) - 1) <= 1u )
      {
        v5 = *(_QWORD **)(v3 + 176);
        if ( v5 )
          break;
      }
      if ( !v2 )
        goto LABEL_10;
    }
    do
    {
      v6 = v5[1];
      do
      {
        v7 = v6;
        v6 = *(_QWORD *)(v6 + 112);
        result = sub_7362F0(v7, 0);
      }
      while ( v6 );
      v5 = (_QWORD *)*v5;
    }
    while ( v5 );
  }
LABEL_10:
  *(_QWORD *)(a1 + 144) = 0;
  return result;
}
