// Function: sub_822C60
// Address: 0x822c60
//
__int64 __fastcall sub_822C60(void *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rbx
  __int64 result; // rax

  if ( !a1 )
    return sub_822BE0(a3, a2, a3, a4, a5, a6);
  v7 = (_QWORD *)qword_4F195F0;
  if ( qword_4F195F0 )
  {
    do
    {
      if ( a1 == (void *)v7[1] )
        break;
      v7 = (_QWORD *)*v7;
    }
    while ( v7 );
  }
  result = realloc(a1);
  if ( !result )
    sub_685240(4u);
  v7[1] = result;
  v7[2] = a3;
  return result;
}
