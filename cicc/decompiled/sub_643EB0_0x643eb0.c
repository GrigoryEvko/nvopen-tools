// Function: sub_643EB0
// Address: 0x643eb0
//
__int64 __fastcall sub_643EB0(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 *v3; // r12
  __int64 v4; // rcx
  void (__fastcall *v5)(__int64); // rdx
  __int64 v6; // rcx
  __int64 v7; // [rsp+8h] [rbp-28h] BYREF

  result = *(_QWORD *)(a1 + 432);
  *(_QWORD *)(a1 + 432) = 0;
  v7 = result;
  if ( result )
  {
    v3 = &v7;
    while ( 1 )
    {
      while ( 1 )
      {
        v5 = *(void (__fastcall **)(__int64))(result + 8);
        if ( a2 )
          break;
        *v3 = *(_QWORD *)result;
        v4 = qword_4CFDE70;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)result = v4;
        qword_4CFDE70 = result;
        *(_BYTE *)(a1 + 132) |= 0x10u;
LABEL_4:
        v5(a1);
        result = *v3;
        if ( !*v3 )
          goto LABEL_8;
      }
      if ( (*(_BYTE *)(result + 16) & 1) == 0 )
      {
        *v3 = *(_QWORD *)result;
        v6 = qword_4CFDE70;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)result = v6;
        qword_4CFDE70 = result;
        goto LABEL_4;
      }
      v3 = (__int64 *)result;
      v5(a1);
      result = *v3;
      if ( !*v3 )
      {
LABEL_8:
        result = v7;
        break;
      }
    }
  }
  *(_QWORD *)(a1 + 432) = result;
  return result;
}
