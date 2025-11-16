// Function: sub_2EAA010
// Address: 0x2eaa010
//
void __fastcall sub_2EAA010(__int64 a1)
{
  unsigned __int64 v1; // r12
  void (__fastcall *v2)(unsigned __int64); // rax

  sub_E66E10(a1 + 8);
  v1 = *(_QWORD *)(a1 + 2496);
  if ( !v1 )
    goto LABEL_4;
  v2 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v1 + 8LL);
  if ( v2 == sub_2EA9B20 )
  {
    nullsub_1605();
    j_j___libc_free_0(v1);
LABEL_4:
    *(_QWORD *)(a1 + 2496) = 0;
    return;
  }
  v2(*(_QWORD *)(a1 + 2496));
  *(_QWORD *)(a1 + 2496) = 0;
}
