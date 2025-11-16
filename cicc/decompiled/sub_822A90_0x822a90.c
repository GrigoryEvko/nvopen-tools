// Function: sub_822A90
// Address: 0x822a90
//
_QWORD *sub_822A90()
{
  __int64 v0; // rbx
  __int64 v1; // rax

  if ( qword_4F07398 > 0 )
  {
    v0 = 0;
    do
    {
      v1 = v0++;
      sub_721990(*(void **)(unk_4F07380 + 16 * v1), *(_QWORD *)(unk_4F07380 + 16 * v1 + 8));
    }
    while ( qword_4F07398 > v0 );
  }
  qword_4F07398 = 0;
  qword_4F195C0 = 0;
  qword_4F07388 = 0;
  qword_4F195B8 = 0;
  return &qword_4F07388;
}
