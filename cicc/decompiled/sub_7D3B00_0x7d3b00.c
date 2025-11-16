// Function: sub_7D3B00
// Address: 0x7d3b00
//
void __fastcall sub_7D3B00(__int64 *a1, __int64 *a2, _QWORD *a3, _QWORD *a4)
{
  char v4; // al
  __int64 *v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v7[0] = a1;
  if ( a1 )
  {
    v4 = *((_BYTE *)a1 + 8);
    if ( v4 != 3 )
      goto LABEL_3;
    sub_72F220(v7);
    a1 = v7[0];
    if ( v7[0] )
    {
      v4 = *((_BYTE *)v7[0] + 8);
LABEL_3:
      if ( v4 )
      {
LABEL_4:
        if ( v4 == 2 )
          sub_7CF470(a1[4], a3, a4);
        a1 = (__int64 *)*v7[0];
        v7[0] = a1;
        if ( a1 )
          goto LABEL_7;
      }
      else
      {
        while ( 1 )
        {
          sub_7D38C0(a1[4], a2);
          a1 = (__int64 *)*v7[0];
          v7[0] = a1;
          if ( !a1 )
            break;
LABEL_7:
          v4 = *((_BYTE *)a1 + 8);
          if ( v4 != 3 )
            goto LABEL_3;
          sub_72F220(v7);
          a1 = v7[0];
          if ( !v7[0] )
            return;
          v4 = *((_BYTE *)v7[0] + 8);
          if ( v4 )
            goto LABEL_4;
        }
      }
    }
  }
}
