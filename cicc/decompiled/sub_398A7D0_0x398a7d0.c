// Function: sub_398A7D0
// Address: 0x398a7d0
//
void (*__fastcall sub_398A7D0(__int64 a1, __int64 a2))()
{
  if ( *(_BYTE *)(a1 + 4502) )
    return (void (*)())sub_396F390(
                         *(_QWORD *)(a1 + 8),
                         *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8LL),
                         *(unsigned int *)(a2 + 64),
                         *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 240LL) + 8LL),
                         0);
  else
    return sub_397C410(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 624), 0);
}
