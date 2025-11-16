// Function: sub_5E4BC0
// Address: 0x5e4bc0
//
__int64 __fastcall sub_5E4BC0(__int64 a1, __int64 a2)
{
  __int64 i; // rax
  __int64 result; // rax
  __int64 v4; // r13

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = **(_QWORD **)(i + 168);
  if ( result && (!*(_QWORD *)result || (*(_BYTE *)(*(_QWORD *)result + 32LL) & 4) != 0) )
  {
    v4 = *(_QWORD *)(result + 8);
    if ( (unsigned int)sub_8D2FB0(v4) )
      v4 = sub_8D46C0(v4);
    result = sub_8D3C40(v4);
    if ( (_DWORD)result )
    {
      *(_BYTE *)(a1 + 194) |= 0x10u;
      result = *(_QWORD *)(a2 + 168);
      *(_BYTE *)(result + 110) |= 2u;
    }
  }
  return result;
}
