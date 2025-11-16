// Function: sub_AD64C0
// Address: 0xad64c0
//
__int64 __fastcall sub_AD64C0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // rbx
  unsigned __int8 *v4; // rsi
  int v5; // edx
  __int64 v7; // [rsp+8h] [rbp-18h]

  v3 = a1;
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
    a1 = **(_QWORD **)(a1 + 16);
  v4 = (unsigned __int8 *)sub_ACD640(a1, a2, a3);
  v5 = *(unsigned __int8 *)(v3 + 8);
  if ( (unsigned int)(v5 - 17) > 1 )
    return (__int64)v4;
  BYTE4(v7) = (_BYTE)v5 == 18;
  LODWORD(v7) = *(_DWORD *)(v3 + 32);
  return sub_AD5E10(v7, v4);
}
