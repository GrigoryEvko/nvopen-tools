// Function: sub_2462030
// Address: 0x2462030
//
__int64 __fastcall sub_2462030(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v3; // rax
  __int64 v5; // [rsp+8h] [rbp-18h]

  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 > 1 )
    return sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL), a3, 0);
  v3 = (unsigned __int8 *)sub_2462030(a1, *(_QWORD *)(a2 + 24));
  BYTE4(v5) = *(_BYTE *)(a2 + 8) == 18;
  LODWORD(v5) = *(_DWORD *)(a2 + 32);
  return sub_AD5E10(v5, v3);
}
