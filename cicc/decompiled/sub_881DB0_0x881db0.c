// Function: sub_881DB0
// Address: 0x881db0
//
void __fastcall sub_881DB0(__int64 a1)
{
  int v2; // edx
  __int64 v3; // rcx
  __int64 i; // rax
  _QWORD *v5; // r12
  char v6; // di
  __int64 v7; // rax

  sub_8790E0((__int64 *)a1);
  if ( (*(_BYTE *)(a1 + 81) & 0x20) == 0 )
  {
    v2 = *(_DWORD *)(a1 + 40);
    v3 = *(_QWORD *)(a1 + 16);
    if ( v2 == -1 )
    {
      if ( a1 == qword_4D04970 )
        qword_4D04970 = *(_QWORD *)(a1 + 16);
      else
        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 24) = *(_QWORD *)(a1 + 24);
      if ( a1 == qword_4F600B8 )
        qword_4F600B8 = *(_QWORD *)(a1 + 24);
    }
    else
    {
      for ( i = qword_4F04C68[0] + 776LL * dword_4F04C64; v2 != *(_DWORD *)i; i -= 776 )
        ;
      v5 = *(_QWORD **)(i + 24);
      v6 = *(_BYTE *)(i + 4);
      if ( !v5 )
        v5 = (_QWORD *)(i + 32);
      if ( a1 == *v5 )
        *v5 = v3;
      else
        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL) = v3;
      v7 = *(_QWORD *)(a1 + 16);
      if ( v7 )
        *(_QWORD *)(v7 + 24) = *(_QWORD *)(a1 + 24);
      if ( a1 == v5[2] )
        v5[2] = *(_QWORD *)(a1 + 24);
      if ( (unsigned int)sub_8770E0(v6) )
        sub_881D30((__int64 *)a1, (__int64)v5);
    }
  }
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
}
