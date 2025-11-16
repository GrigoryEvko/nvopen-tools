// Function: sub_824020
// Address: 0x824020
//
__int64 __fastcall sub_824020(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r12
  _QWORD *v6; // rbx
  unsigned __int64 v7; // r12

  qword_4F061C8 = 0;
  if ( qword_4F073B0 )
    sub_823420(a1, a2);
  if ( dword_4F195C8 )
    sub_7219E0();
  result = (__int64)sub_822A90();
  v3 = (_QWORD *)qword_4F195E8;
  if ( qword_4F195E8 )
  {
    do
    {
      result = _libc_free(v3[4], a2);
      v3[4] = 0;
      v3 = (_QWORD *)*v3;
    }
    while ( v3 );
  }
  v4 = (_QWORD *)qword_4F195F8;
  qword_4F195E8 = 0;
  if ( qword_4F195F8 )
  {
    do
    {
      v5 = (unsigned __int64)v4;
      v4 = (_QWORD *)*v4;
      result = _libc_free(*(_QWORD *)(v5 + 8), a2);
      if ( v5 >= (unsigned __int64)&qword_4F1F620 || v5 < (unsigned __int64)(&qword_4F1F620 - 3072) )
        result = _libc_free(v5, a2);
    }
    while ( v4 );
  }
  v6 = (_QWORD *)qword_4F195F0;
  qword_4F195F8 = 0;
  if ( qword_4F195F0 )
  {
    do
    {
      v7 = (unsigned __int64)v6;
      v6 = (_QWORD *)*v6;
      result = _libc_free(*(_QWORD *)(v7 + 8), a2);
      if ( v7 >= (unsigned __int64)&qword_4F1F620 || v7 < (unsigned __int64)(&qword_4F1F620 - 3072) )
        result = _libc_free(v7, a2);
    }
    while ( v6 );
  }
  qword_4F195F0 = 0;
  return result;
}
