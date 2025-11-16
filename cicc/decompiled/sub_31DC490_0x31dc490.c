// Function: sub_31DC490
// Address: 0x31dc490
//
__int64 __fastcall sub_31DC490(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 (__fastcall *v6)(__int64, __int64, __int64); // r14
  __int64 v7; // [rsp-40h] [rbp-40h]

  result = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
  {
    v3 = a2;
    v4 = 0;
    v7 = (unsigned int)(result - 1);
    while ( 1 )
    {
      result = (__int64)sub_BD3990(*(unsigned __int8 **)(v3 + 32 * (v4 - (unsigned int)result)), a2);
      a2 = result;
      if ( *(_BYTE *)result <= 3u )
      {
        v5 = *(_QWORD *)(a1 + 224);
        v6 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v5 + 296LL);
        a2 = sub_31DB510(a1, result);
        result = v6(v5, a2, 18);
      }
      if ( v4 == v7 )
        break;
      ++v4;
      LODWORD(result) = *(_DWORD *)(v3 + 4) & 0x7FFFFFF;
    }
  }
  return result;
}
