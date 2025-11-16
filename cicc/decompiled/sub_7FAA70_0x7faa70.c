// Function: sub_7FAA70
// Address: 0x7faa70
//
void __fastcall sub_7FAA70(__int64 a1)
{
  __int64 v1; // r14
  __int64 *v2; // r13
  __int64 **v3; // rbx
  __int64 *v4; // r15
  __int64 *v5; // r12
  __int64 *v6; // rax
  __int64 v7; // r11
  __m128i *v8; // rax
  __int64 *v9; // rax
  __int64 *i; // r9
  __int64 *v11; // rcx
  __int64 v12; // [rsp+0h] [rbp-60h]
  __int64 v13; // [rsp+10h] [rbp-50h]
  __int64 *v14; // [rsp+28h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 88);
  v14 = 0;
  if ( !v1 )
  {
    v3 = *(__int64 ***)(a1 + 48);
    if ( !v3 )
      return;
    v2 = 0;
    goto LABEL_3;
  }
  v2 = *(__int64 **)(v1 + 24);
  *(_QWORD *)(v1 + 24) = 0;
  v3 = *(__int64 ***)(a1 + 48);
  v14 = v2;
  if ( v3 )
  {
LABEL_3:
    v13 = 0;
    v4 = 0;
    v5 = 0;
    do
    {
      if ( ((_WORD)v3[1] & 0x8FF) == 0x802 )
      {
        v7 = v3[2][19];
        if ( !v1 )
        {
          v13 = qword_4F06BC0;
          v1 = *(_QWORD *)(a1 + 88);
          if ( !v1 )
          {
            v12 = v3[2][19];
            qword_4F06BC0 = *(_QWORD *)(qword_4F07288 + 88);
            sub_733780(0x17u, a1, 0, 1, 0);
            v1 = *(_QWORD *)(a1 + 88);
            v7 = v12;
          }
          qword_4F06BC0 = v1;
        }
        if ( v2 )
        {
          if ( v4 == v2 )
          {
            if ( v5 )
              *(_QWORD *)(*v5 + 32) = *(_QWORD *)(v1 + 24);
            v14 = 0;
            v2 = 0;
            v5 = 0;
          }
          else
          {
            v9 = (__int64 *)v2[4];
            if ( v4 == v9 )
            {
              v5 = (__int64 *)&v14;
              v11 = v2;
            }
            else
            {
              for ( i = v2; ; i = v11 )
              {
                v11 = v9;
                v9 = (__int64 *)v9[4];
                if ( v9 == v4 )
                  break;
              }
              v5 = i + 4;
            }
            v11[4] = 0;
          }
          *(_QWORD *)(v1 + 24) = v4;
        }
        v8 = sub_740B80(v7, 0x8000u);
        v3[3] = (__int64 *)v8;
        v8[3].m128i_i8[1] |= 0x20u;
        if ( !v8[1].m128i_i64[1] && v8[1].m128i_i64[0] )
          sub_732D90((__int64)v8, v1);
        if ( v5 )
          *(_QWORD *)(*v5 + 32) = *(_QWORD *)(v1 + 24);
      }
      v6 = v3[3];
      if ( v6 && v6[2] && v6[3] == v1 )
        v4 = v3[3];
      v3 = (__int64 **)*v3;
    }
    while ( v3 );
    if ( v2 )
    {
      if ( v5 )
        *(_QWORD *)(*v5 + 32) = *(_QWORD *)(v1 + 24);
      *(_QWORD *)(v1 + 24) = v2;
    }
    if ( v13 )
    {
      if ( (unsigned int)sub_733920(*(_QWORD *)(a1 + 88)) )
        sub_733650(*(_QWORD *)(a1 + 88));
      qword_4F06BC0 = v13;
    }
    return;
  }
  if ( v2 )
    *(_QWORD *)(v1 + 24) = v2;
}
