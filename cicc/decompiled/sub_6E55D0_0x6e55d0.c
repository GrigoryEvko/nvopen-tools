// Function: sub_6E55D0
// Address: 0x6e55d0
//
__int64 __fastcall sub_6E55D0(unsigned int a1, unsigned int a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // cl
  unsigned __int64 v4; // r9
  unsigned __int64 v5; // r8

  result = a1;
  v3 = a2;
  v4 = dword_4D04120[(unsigned __int8)a1];
  v5 = dword_4D04120[(unsigned __int8)a2];
  if ( v4 <= v5 && (v4 != v5 || (unsigned __int8)a1 <= (unsigned __int8)a2) )
  {
    result = a2;
    v3 = a1;
  }
  if ( (_BYTE)result == 14 || (unsigned __int8)result <= 8u )
  {
    if ( dword_4F077C4 == 1 && (_BYTE)result == 2 )
      return 4;
    return result;
  }
  if ( dword_4D04120[(unsigned __int8)result] != dword_4D04120[v3]
    || dword_4D04020[(unsigned __int8)result] != dword_4D04020[v3] )
  {
    if ( v3 <= 8u || v3 == 14 || (_BYTE)result == v3 || qword_4D040A0[(unsigned __int8)result] != qword_4D040A0[v3] )
      return result;
    goto LABEL_22;
  }
  if ( qword_4D040A0[(unsigned __int8)result] != qword_4D040A0[v3] )
    return result;
  if ( v3 == 2 )
  {
    if ( dword_4D04120[2] != dword_4D04120[4] )
      return result;
    return 4;
  }
  if ( v3 == 4 )
  {
    if ( dword_4D04120[4] != dword_4D04120[6] )
      return result;
    return 4;
  }
  if ( v3 != 14 && v3 > 8u && (_BYTE)result != v3 )
  {
LABEL_22:
    if ( (unsigned int)sub_6E5430() )
      sub_6851C0(0xCB1u, dword_4F07508);
    return 4;
  }
  return result;
}
