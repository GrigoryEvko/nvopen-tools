// Function: sub_34337D0
// Address: 0x34337d0
//
__int64 __fastcall sub_34337D0(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 (*v6)(); // rax

  v2 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned int)(v2 - 17) <= 1 )
    LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(a1 + 16) + 8LL);
  result = 0;
  if ( (_BYTE)v2 == 14 )
  {
    v4 = *(_QWORD *)(a2 + 976);
    if ( v4 && (v5 = *(_QWORD *)(v4 + 8), v6 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL), v6 != sub_BD8D60) )
    {
      result = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a1);
      if ( !BYTE1(result) )
        return 1;
    }
    else
    {
      return 1;
    }
  }
  return result;
}
