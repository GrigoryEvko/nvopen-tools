// Function: sub_9415C0
// Address: 0x9415c0
//
void __fastcall sub_9415C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  int v7; // r9d
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 *v15; // rdi
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_93ED80(*(_DWORD *)(a1 + 448), (char *)v19);
  v8 = 0;
  v9 = sub_9405D0(a1, *(_DWORD *)(a1 + 448), v4, v5, v6, v7);
  v10 = *(_QWORD *)(a1 + 512);
  if ( *(_QWORD *)(a1 + 480) != v10 )
  {
    if ( v10 == *(_QWORD *)(a1 + 520) )
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 536) - 8LL) + 512LL;
    v8 = *(_QWORD *)(v10 - 8);
  }
  v11 = sub_ADD770(a1 + 16, v8, v9, v19[0], *(unsigned __int16 *)(a1 + 452));
  v12 = *(__int64 **)(a1 + 512);
  v13 = v11;
  if ( v12 == (__int64 *)(*(_QWORD *)(a1 + 528) - 8LL) )
  {
    v14 = *(_QWORD *)(a1 + 536);
    if ( (((__int64)v12 - *(_QWORD *)(a1 + 520)) >> 3)
       + ((((v14 - *(_QWORD *)(a1 + 504)) >> 3) - 1) << 6)
       + ((__int64)(*(_QWORD *)(a1 + 496) - *(_QWORD *)(a1 + 480)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 472) - ((v14 - *(_QWORD *)(a1 + 464)) >> 3)) <= 1 )
    {
      sub_93FFB0((__int64 *)(a1 + 464), 1u, 0);
      v14 = *(_QWORD *)(a1 + 536);
    }
    *(_QWORD *)(v14 + 8) = sub_22077B0(512);
    v15 = *(__int64 **)(a1 + 512);
    if ( v15 )
    {
      *v15 = v13;
      if ( v13 )
        sub_B96E90(v15, v13, 1);
    }
    v16 = (__int64 *)(*(_QWORD *)(a1 + 536) + 8LL);
    *(_QWORD *)(a1 + 536) = v16;
    v17 = *v16;
    v18 = *v16 + 512;
    *(_QWORD *)(a1 + 520) = v17;
    *(_QWORD *)(a1 + 528) = v18;
    *(_QWORD *)(a1 + 512) = v17;
  }
  else
  {
    if ( v12 )
    {
      *v12 = v11;
      if ( v11 )
        sub_B96E90(v12, v11, 1);
      v12 = *(__int64 **)(a1 + 512);
    }
    *(_QWORD *)(a1 + 512) = v12 + 1;
  }
  sub_93FCC0(a1, a2);
}
