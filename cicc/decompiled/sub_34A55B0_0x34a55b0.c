// Function: sub_34A55B0
// Address: 0x34a55b0
//
_QWORD *__fastcall sub_34A55B0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rcx
  _QWORD *v5; // rax
  char v6; // si
  unsigned __int64 v7; // rdx
  __int64 v8; // rax

  v3 = *(_QWORD **)(a1 + 16);
  if ( v3 )
  {
    v4 = *a2;
    while ( 1 )
    {
      v7 = v3[4];
      if ( v7 > v4 || v7 == v4 && a2[1] < v3[5] )
      {
        v5 = (_QWORD *)v3[2];
        v6 = 1;
        if ( !v5 )
        {
LABEL_9:
          if ( v6 )
            goto LABEL_10;
LABEL_12:
          if ( v7 < v4 || v7 == v4 && v3[5] < a2[1] )
            return 0;
          return v3;
        }
      }
      else
      {
        v5 = (_QWORD *)v3[3];
        v6 = 0;
        if ( !v5 )
          goto LABEL_9;
      }
      v3 = v5;
    }
  }
  v3 = (_QWORD *)(a1 + 8);
LABEL_10:
  if ( *(_QWORD **)(a1 + 24) != v3 )
  {
    v8 = sub_220EF80((__int64)v3);
    v4 = *a2;
    v7 = *(_QWORD *)(v8 + 32);
    v3 = (_QWORD *)v8;
    goto LABEL_12;
  }
  return 0;
}
