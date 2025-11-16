// Function: sub_D5B880
// Address: 0xd5b880
//
__int64 *__fastcall sub_D5B880(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v3; // rcx
  __int64 *v4; // rax
  __int64 v5; // r14
  __int64 *v6; // rax
  __int64 **v7; // rdx
  __int64 v8; // rdx

  result = sub_D5B790((_QWORD *)(a1 + 568), a2);
  if ( *(_QWORD *)(a1 + 656) == a2 )
  {
    v3 = *(_QWORD *)(a1 + 632);
    v4 = *(__int64 **)(a1 + 616);
    *(_BYTE *)(a1 + 664) = 1;
    if ( v4 == (__int64 *)(v3 - 8) )
    {
      v5 = *(_QWORD *)(a1 + 640);
      if ( (((__int64)v4 - *(_QWORD *)(a1 + 624)) >> 3)
         + ((((v5 - *(_QWORD *)(a1 + 608)) >> 3) - 1) << 6)
         + ((__int64)(*(_QWORD *)(a1 + 600) - *(_QWORD *)(a1 + 584)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 576) - ((v5 - *(_QWORD *)(a1 + 568)) >> 3)) <= 1 )
      {
        sub_D58C10((__int64 *)(a1 + 568), 1u, 0);
        v5 = *(_QWORD *)(a1 + 640);
      }
      *(_QWORD *)(v5 + 8) = sub_22077B0(512);
      v6 = *(__int64 **)(a1 + 616);
      if ( v6 )
        *v6 = a2;
      v7 = (__int64 **)(*(_QWORD *)(a1 + 640) + 8LL);
      *(_QWORD *)(a1 + 640) = v7;
      result = *v7;
      v8 = (__int64)(*v7 + 64);
      *(_QWORD *)(a1 + 624) = result;
      *(_QWORD *)(a1 + 632) = v8;
      *(_QWORD *)(a1 + 616) = result;
    }
    else
    {
      if ( v4 )
      {
        *v4 = a2;
        v4 = *(__int64 **)(a1 + 616);
      }
      result = v4 + 1;
      *(_QWORD *)(a1 + 616) = result;
    }
  }
  return result;
}
