// Function: sub_5D0F70
// Address: 0x5d0f70
//
__int64 __fastcall sub_5D0F70(__int64 a1)
{
  char v1; // al
  __int64 result; // rax
  __int64 v3; // rsi

  v1 = *(_BYTE *)(a1 + 80);
  if ( v1 == 7 )
  {
    result = *(_QWORD *)(a1 + 88);
    if ( (*(_BYTE *)(result + 169) & 8) == 0 )
      return result;
    v3 = *(_QWORD *)(result + 144);
    if ( !v3 )
      return result;
    goto LABEL_4;
  }
  if ( v1 != 11 )
    sub_721090(a1);
  result = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 256LL);
  v3 = *(_QWORD *)(result + 40);
  if ( v3 )
  {
LABEL_4:
    result = sub_881B20(qword_4CF6E18, v3, 1);
    if ( !*(_QWORD *)result || (*(_QWORD *)(*(_QWORD *)result + 80LL) & 0x800000200LL) == 0 )
      *(_QWORD *)result = a1;
  }
  return result;
}
