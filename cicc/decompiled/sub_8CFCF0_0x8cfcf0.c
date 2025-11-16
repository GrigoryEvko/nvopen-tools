// Function: sub_8CFCF0
// Address: 0x8cfcf0
//
unsigned int *sub_8CFCF0()
{
  __int64 *v1; // r12
  _QWORD *v2; // r12
  _QWORD *v3; // rbx

  if ( !dword_4D03FE8[0] && qword_4F074B0 == qword_4F60258 )
  {
    unk_4D03FC4 = 1;
    v1 = *(__int64 **)(qword_4D03FF0 + 8);
    sub_8CFAC0(v1);
    sub_8CF7C0(v1);
    v2 = (_QWORD *)qword_4F60250;
    if ( qword_4F60250 )
    {
LABEL_5:
      qword_4F60250 = 0;
      while ( 2 )
      {
        v3 = v2;
        v2 = (_QWORD *)*v2;
        switch ( *((_BYTE *)v3 + 8) )
        {
          case 0:
            goto LABEL_8;
          case 2:
            sub_8C77C0(v3[2]);
            *v3 = qword_4F60248;
            if ( !v2 )
              goto LABEL_9;
            continue;
          case 6:
            sub_8CD5A0((__int64 *)v3[2]);
LABEL_8:
            *v3 = qword_4F60248;
            if ( !v2 )
              goto LABEL_9;
            continue;
          case 7:
            sub_8C7A50(v3[2]);
            *v3 = qword_4F60248;
            if ( !v2 )
              goto LABEL_9;
            continue;
          case 8:
            sub_8C78D0(v3[2]);
            *v3 = qword_4F60248;
            if ( !v2 )
              goto LABEL_9;
            continue;
          case 0xB:
            sub_8CDA30(v3[2]);
            *v3 = qword_4F60248;
            if ( !v2 )
              goto LABEL_9;
            continue;
          case 0x1C:
            sub_8C7150(v3[2]);
            *v3 = qword_4F60248;
            if ( !v2 )
              goto LABEL_9;
            continue;
          case 0x3B:
            sub_8CE3E0(v3[2]);
            *v3 = qword_4F60248;
            if ( v2 )
              continue;
LABEL_9:
            v2 = (_QWORD *)qword_4F60250;
            if ( !qword_4F60250 )
              goto LABEL_10;
            goto LABEL_5;
          default:
            sub_721090();
        }
      }
    }
LABEL_10:
    unk_4D03FC4 = 0;
    dword_4D03FC0 = 1;
    return &dword_4D03FC0;
  }
  else
  {
    dword_4D03FC0 = 1;
    return &dword_4D03FC0;
  }
}
