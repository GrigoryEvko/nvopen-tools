// Function: sub_7F5F50
// Address: 0x7f5f50
//
unsigned __int64 __fastcall sub_7F5F50(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 result; // rax
  __int64 *v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rax
  unsigned __int64 v7; // rbx

  v2 = a1;
  if ( (unsigned int)sub_8D3410(a1) )
    v2 = sub_8D40F0(a1);
  if ( !(unsigned int)sub_691630(v2, 0) )
    return 0;
  if ( a2 )
  {
    v4 = sub_7E5340(a2);
    if ( (*(_BYTE *)(a2 + 89) & 4) == 0 )
    {
      v5 = *(_QWORD *)(a2 + 40);
      if ( !v5 || *(_BYTE *)(v5 + 28) != 3 )
      {
        v6 = (_QWORD *)*v4;
        if ( v6 )
        {
          if ( !*v6 && (unsigned int)sub_8D4C80(v6[1]) )
            return 0;
        }
      }
    }
  }
  v7 = sub_72BA30(byte_4F06A51[0])[16];
  if ( dword_4F06880 )
    v7 *= 2LL;
  if ( *(char *)(v2 + 142) >= 0 && *(_BYTE *)(v2 + 140) == 12 )
    LODWORD(result) = sub_8D4AB0(v2, 0, dword_4F06880);
  else
    LODWORD(result) = *(_DWORD *)(v2 + 136);
  result = (unsigned int)result;
  if ( (unsigned int)result < v7 )
    return v7;
  return result;
}
