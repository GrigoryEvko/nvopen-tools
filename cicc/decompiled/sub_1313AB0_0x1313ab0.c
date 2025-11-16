// Function: sub_1313AB0
// Address: 0x1313ab0
//
void __fastcall sub_1313AB0(__int64 a1, unsigned __int8 a2)
{
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 816) <= 2u )
  {
    if ( a2 > 2u )
    {
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
      if ( a1 == qword_4F96B50 )
      {
        if ( a1 == *(_QWORD *)(a1 + 200) )
        {
          qword_4F96B50 = 0;
          goto LABEL_9;
        }
        qword_4F96B50 = *(_QWORD *)(a1 + 200);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 208) + 200LL) = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 208LL);
      v2 = *(_QWORD *)(a1 + 208);
      *(_QWORD *)(*(_QWORD *)(a1 + 200) + 208LL) = v2;
      *(_QWORD *)(a1 + 208) = *(_QWORD *)(v2 + 200);
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 200) + 208LL) + 200LL) = *(_QWORD *)(a1 + 200);
      *(_QWORD *)(*(_QWORD *)(a1 + 208) + 200LL) = a1;
LABEL_9:
      byte_4F96B48 = 0;
      pthread_mutex_unlock(&stru_4F96B20);
      *(_BYTE *)(a1 + 816) = a2;
      sub_1313270(a1);
      return;
    }
    sub_1313A40((_BYTE *)a1);
    goto LABEL_11;
  }
  *(_BYTE *)(a1 + 816) = a2;
  if ( a2 > 2u )
  {
LABEL_11:
    sub_1313270(a1);
    return;
  }
  sub_1313920(a1);
  sub_1313270(a1);
}
