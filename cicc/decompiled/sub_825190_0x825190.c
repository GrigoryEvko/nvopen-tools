// Function: sub_825190
// Address: 0x825190
//
_QWORD *sub_825190()
{
  _QWORD *result; // rax
  _QWORD *v1; // rbx
  _BYTE *v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 *v6; // r9
  __int64 v7; // rdi

  result = &qword_4F07280;
  v1 = (_QWORD *)qword_4F07320[0];
  if ( qword_4F07320[0] )
  {
    do
    {
      result = (_QWORD *)v1[4];
      v2 = (_BYTE *)result[4];
      if ( v2 )
      {
        sub_824E90(v2);
        result = (_QWORD *)v1[4];
        v7 = result[4];
        if ( v7 )
        {
          sub_823A00(v7, 64, v3, v4, v5, v6);
          result = (_QWORD *)v1[4];
        }
        result[4] = 0;
      }
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
  }
  return result;
}
