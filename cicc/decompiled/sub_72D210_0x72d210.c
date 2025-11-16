// Function: sub_72D210
// Address: 0x72d210
//
__int64 sub_72D210()
{
  __int64 result; // rax
  _QWORD *v1; // rsi
  __int64 v2; // rdi
  __int64 *v3; // rax
  _QWORD *v4; // rcx
  __int64 *v5; // rdx

  result = unk_4D03FF0;
  v1 = *(_QWORD **)(unk_4D03FF0 + 360LL);
  if ( v1 )
  {
    do
    {
      v2 = v1[1];
      v3 = *(__int64 **)(v2 + 120);
      if ( v3 )
      {
        v4 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v5 = (__int64 *)*v3;
            if ( *((_BYTE *)v3 + 17) )
              break;
            v4 = v3;
            v3 = (__int64 *)*v3;
            if ( !v5 )
              goto LABEL_9;
          }
          if ( !v4 )
            break;
          *v4 = v5;
          v3 = v5;
LABEL_6:
          if ( !v3 )
            goto LABEL_9;
        }
        while ( 1 )
        {
          *(_QWORD *)(v2 + 120) = v5;
          if ( !v5 )
            break;
          v3 = (__int64 *)*v5;
          if ( !*((_BYTE *)v5 + 17) )
          {
            v4 = v5;
            goto LABEL_6;
          }
          v5 = (__int64 *)*v5;
        }
      }
LABEL_9:
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
    result = unk_4D03FF0;
  }
  *(_QWORD *)(result + 360) = 0;
  return result;
}
