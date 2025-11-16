// Function: sub_2DAC8E0
// Address: 0x2dac8e0
//
__int64 __fastcall sub_2DAC8E0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 (*v2)(); // rdx
  __int64 (*v3)(); // rdx
  __int64 (*v4)(); // rax

  v1 = *a1;
  v2 = *(__int64 (**)())(*a1 + 160LL);
  if ( v2 == sub_2DAC7A0 )
  {
    a1[22] = 0;
    v3 = *(__int64 (**)())(v1 + 168);
    if ( v3 == sub_2DAC7B0 )
      goto LABEL_3;
LABEL_6:
    a1[23] = ((__int64 (__fastcall *)(_QWORD *))v3)(a1);
    v4 = *(__int64 (**)())(*a1 + 176LL);
    if ( v4 == sub_2DAC7C0 )
      goto LABEL_4;
    goto LABEL_7;
  }
  a1[22] = v2();
  v1 = *a1;
  v3 = *(__int64 (**)())(*a1 + 168LL);
  if ( v3 != sub_2DAC7B0 )
    goto LABEL_6;
LABEL_3:
  a1[23] = 0;
  v4 = *(__int64 (**)())(v1 + 176);
  if ( v4 == sub_2DAC7C0 )
  {
LABEL_4:
    a1[24] = 0;
    return 0;
  }
LABEL_7:
  a1[24] = ((__int64 (__fastcall *)(_QWORD *))v4)(a1);
  return 0;
}
