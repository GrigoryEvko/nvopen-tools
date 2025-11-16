// Function: sub_1313870
// Address: 0x1313870
//
int __fastcall sub_1313870(__int64 a1)
{
  __int64 v1; // rax

  if ( pthread_mutex_trylock(&stru_4F96B20) )
  {
    sub_130AD90((__int64)&unk_4F96AE0);
    byte_4F96B48 = 1;
  }
  ++qword_4F96B18;
  if ( a1 != qword_4F96B10 )
  {
    ++qword_4F96B08;
    qword_4F96B10 = a1;
  }
  v1 = qword_4F96B50;
  do
  {
    if ( !v1 )
      break;
    *(_BYTE *)(v1 + 816) = 2;
    _mm_mfence();
    *(_QWORD *)(v1 + 832) = 0;
    *(_QWORD *)(v1 + 848) = 0;
    v1 = *(_QWORD *)(v1 + 200);
  }
  while ( v1 != qword_4F96B50 );
  byte_4F96B48 = 0;
  return pthread_mutex_unlock(&stru_4F96B20);
}
