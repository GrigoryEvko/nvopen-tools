// Function: sub_1313920
// Address: 0x1313920
//
int __fastcall sub_1313920(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx

  *(_QWORD *)(a1 + 200) = a1;
  *(_QWORD *)(a1 + 208) = a1;
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
  if ( qword_4F96B50 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 208) + 200LL) = *(_QWORD *)(qword_4F96B50 + 208);
    v2 = *(_QWORD *)(a1 + 208);
    *(_QWORD *)(v1 + 208) = v2;
    *(_QWORD *)(a1 + 208) = *(_QWORD *)(v2 + 200);
    *(_QWORD *)(*(_QWORD *)(v1 + 208) + 200LL) = v1;
    *(_QWORD *)(*(_QWORD *)(a1 + 208) + 200LL) = a1;
  }
  qword_4F96B50 = *(_QWORD *)(a1 + 200);
  byte_4F96B48 = 0;
  return pthread_mutex_unlock(&stru_4F96B20);
}
