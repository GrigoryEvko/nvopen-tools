// Function: sub_12A0660
// Address: 0x12a0660
//
void __fastcall sub_12A0660(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 *v11; // rdi
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_129E300(*(_DWORD *)(a1 + 480), (char *)v15);
  v4 = 0;
  v5 = sub_129F850(a1, *(_DWORD *)(a1 + 480));
  v6 = *(_QWORD *)(a1 + 544);
  if ( *(_QWORD *)(a1 + 512) != v6 )
  {
    if ( v6 == *(_QWORD *)(a1 + 552) )
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 568) - 8LL) + 512LL;
    v4 = *(_QWORD *)(v6 - 8);
  }
  v7 = sub_15A69E0(a1 + 16, v4, v5, v15[0], *(unsigned __int16 *)(a1 + 484));
  v8 = *(__int64 **)(a1 + 544);
  v9 = v7;
  if ( v8 == (__int64 *)(*(_QWORD *)(a1 + 560) - 8LL) )
  {
    v10 = *(_QWORD *)(a1 + 568);
    if ( (((__int64)v8 - *(_QWORD *)(a1 + 552)) >> 3)
       + ((((v10 - *(_QWORD *)(a1 + 536)) >> 3) - 1) << 6)
       + ((__int64)(*(_QWORD *)(a1 + 528) - *(_QWORD *)(a1 + 512)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 504) - ((v10 - *(_QWORD *)(a1 + 496)) >> 3)) <= 1 )
    {
      sub_129F230((__int64 *)(a1 + 496), 1u, 0);
      v10 = *(_QWORD *)(a1 + 568);
    }
    *(_QWORD *)(v10 + 8) = sub_22077B0(512);
    v11 = *(__int64 **)(a1 + 544);
    if ( v11 )
    {
      *v11 = v9;
      if ( v9 )
        sub_1623A60(v11, v9, 2);
    }
    v12 = (__int64 *)(*(_QWORD *)(a1 + 568) + 8LL);
    *(_QWORD *)(a1 + 568) = v12;
    v13 = *v12;
    v14 = *v12 + 512;
    *(_QWORD *)(a1 + 552) = v13;
    *(_QWORD *)(a1 + 560) = v14;
    *(_QWORD *)(a1 + 544) = v13;
  }
  else
  {
    if ( v8 )
    {
      *v8 = v7;
      if ( v7 )
        sub_1623A60(v8, v7, 2);
      v8 = *(__int64 **)(a1 + 544);
    }
    *(_QWORD *)(a1 + 544) = v8 + 1;
  }
  sub_129F080(a1, a2);
}
