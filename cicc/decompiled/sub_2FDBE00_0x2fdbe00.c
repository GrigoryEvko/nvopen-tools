// Function: sub_2FDBE00
// Address: 0x2fdbe00
//
__int64 __fastcall sub_2FDBE00(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 (*v5)(); // rdx

  if ( (unsigned __int8)sub_B2D610(*(_QWORD *)a2, 20) )
    return 0;
  v2 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 392LL))(a1, a2);
  if ( !(_BYTE)v2 )
    return 0;
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v4 = *(_QWORD *)v3;
  v5 = *(__int64 (**)())(*(_QWORD *)v3 + 488LL);
  if ( v5 == sub_2FDBB50 )
    goto LABEL_4;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v5)(v3, a2) )
    return 0;
  v4 = *(_QWORD *)v3;
LABEL_4:
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64))(v4 + 544))(v3, a2) )
    return (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)v3 + 536LL))(v3, a2) ^ 1;
  return v2;
}
