// Function: sub_5CD370
// Address: 0x5cd370
//
__int64 __fastcall sub_5CD370(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // r14d
  char *v4; // rax
  __int64 (__fastcall *v5)(__int64, __int64, _QWORD); // rbx

  v3 = a3;
  v4 = (char *)&unk_496EE40 + 24 * *(unsigned __int8 *)(a1 + 8);
  v5 = (__int64 (__fastcall *)(__int64, __int64, _QWORD))*((_QWORD *)v4 + 2);
  if ( (unsigned int)sub_5CCB50(*((char **)v4 + 1), a1, a2, a3) && *(_BYTE *)(a1 + 8) && v5 )
    return v5(a1, a2, v3);
  else
    return a2;
}
