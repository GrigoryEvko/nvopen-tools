// Function: sub_2E393D0
// Address: 0x2e393d0
//
__int64 __fastcall sub_2E393D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v4; // r12
  __int64 *v5; // rax
  __m128i *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-A8h]
  _QWORD v13[20]; // [rsp+10h] [rbp-A0h] BYREF

  v4 = a2;
  v5 = *(__int64 **)(a1 + 32);
  if ( v5 )
  {
    v12 = *v5;
    sub_A558A0((__int64)v13, *(_QWORD *)(*v5 + 40), 1);
    sub_A564B0((__int64)v13, v12);
    sub_2E38390(a1, a2, (__int64)v13, a3, a4);
    return (__int64)sub_A55520(v13, a2);
  }
  else
  {
    v9 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v9 <= 0x3Fu )
    {
      v11 = sub_CB6200(a2, "Can't print out MachineBasicBlock because parent MachineFunction", 0x40u);
      v10 = *(_QWORD *)(v11 + 32);
      v4 = v11;
    }
    else
    {
      *v9 = _mm_load_si128((const __m128i *)&xmmword_42E9C20);
      v9[1] = _mm_load_si128((const __m128i *)&xmmword_42E9C30);
      v9[2] = _mm_load_si128((const __m128i *)&xmmword_42E9C40);
      v9[3] = _mm_load_si128((const __m128i *)&xmmword_42E9C50);
      v10 = *(_QWORD *)(a2 + 32) + 64LL;
      *(_QWORD *)(a2 + 32) = v10;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v10) <= 8 )
    {
      return sub_CB6200(v4, " is null\n", 9u);
    }
    else
    {
      *(_BYTE *)(v10 + 8) = 10;
      *(_QWORD *)v10 = 0x6C6C756E20736920LL;
      *(_QWORD *)(v4 + 32) += 9LL;
      return 0x6C6C756E20736920LL;
    }
  }
}
