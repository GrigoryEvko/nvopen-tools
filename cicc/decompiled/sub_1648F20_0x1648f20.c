// Function: sub_1648F20
// Address: 0x1648f20
//
void __fastcall sub_1648F20(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // r14
  _QWORD *v8; // rax

  v4 = *(_QWORD **)(a1 + 8);
  if ( v4 )
  {
    while ( 1 )
    {
      v7 = (_QWORD *)v4[1];
      v8 = sub_1648700((__int64)v4);
      if ( *((_BYTE *)v8 + 16) <= 0x17u || a3 != v8[5] )
      {
        if ( *v4 )
        {
          v5 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v5 = v7;
          if ( v7 )
          {
            v7[2] = v7[2] & 3LL | v5;
            *v4 = a2;
            if ( !a2 )
              goto LABEL_9;
          }
          else
          {
            *v4 = a2;
            if ( !a2 )
              return;
          }
        }
        else
        {
          *v4 = a2;
          if ( !a2 )
            goto LABEL_8;
        }
        v6 = *(_QWORD *)(a2 + 8);
        v4[1] = v6;
        if ( v6 )
          *(_QWORD *)(v6 + 16) = (unsigned __int64)(v4 + 1) | *(_QWORD *)(v6 + 16) & 3LL;
        v4[2] = (a2 + 8) | v4[2] & 3LL;
        *(_QWORD *)(a2 + 8) = v4;
      }
LABEL_8:
      if ( !v7 )
        return;
LABEL_9:
      v4 = v7;
    }
  }
}
