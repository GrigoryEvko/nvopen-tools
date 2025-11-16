// Function: sub_1648D30
// Address: 0x1648d30
//
_BOOL8 __fastcall sub_1648D30(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  signed __int64 v8; // rdx
  _QWORD *v9; // rdx
  _QWORD *v10; // rax
  bool v12; // [rsp+7h] [rbp-39h]

  v3 = *(_QWORD *)(a2 + 48);
  v4 = *(_QWORD *)(a1 + 8);
  v12 = v4 != 0 && v3 != a2 + 40;
  if ( v12 )
  {
    do
    {
      if ( !v3 )
        BUG();
      v5 = 24LL * (*(_DWORD *)(v3 - 4) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v3 - 1) & 0x40) != 0 )
      {
        v6 = *(_QWORD **)(v3 - 32);
        v7 = &v6[(unsigned __int64)v5 / 8];
      }
      else
      {
        v7 = (_QWORD *)(v3 - 24);
        v6 = (_QWORD *)(v3 - 24 - v5);
      }
      v8 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 3);
      if ( v8 >> 2 )
      {
        v9 = &v6[12 * (v8 >> 2)];
        while ( a1 != *v6 )
        {
          if ( a1 == v6[3] )
          {
            v6 += 3;
            break;
          }
          if ( a1 == v6[6] )
          {
            v6 += 6;
            break;
          }
          if ( a1 == v6[9] )
          {
            v6 += 9;
            break;
          }
          v6 += 12;
          if ( v9 == v6 )
          {
            v8 = 0xAAAAAAAAAAAAAAABLL * (v7 - v6);
            goto LABEL_21;
          }
        }
      }
      else
      {
LABEL_21:
        if ( v8 == 2 )
          goto LABEL_31;
        if ( v8 == 3 )
        {
          if ( a1 != *v6 )
          {
            v6 += 3;
LABEL_31:
            if ( a1 != *v6 )
            {
              v6 += 3;
              if ( a1 != *v6 )
                goto LABEL_13;
            }
          }
        }
        else if ( v8 != 1 || a1 != *v6 )
        {
          goto LABEL_13;
        }
      }
      if ( v7 != v6 )
        return v12;
LABEL_13:
      v10 = sub_1648700(v4);
      if ( *((_BYTE *)v10 + 16) > 0x17u && a2 == v10[5] )
        return v12;
      v4 = *(_QWORD *)(v4 + 8);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v4 && a2 + 40 != v3 );
  }
  return 0;
}
