// Function: sub_396F250
// Address: 0x396f250
//
__int64 __fastcall sub_396F250(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 (__fastcall *v7)(__int64, __int64, __int64); // [rsp-40h] [rbp-40h]

  result = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v3 = 0;
    v4 = (unsigned int)(result - 1);
    while ( 1 )
    {
      result = sub_1649C60(*(_QWORD *)(a2 + 24 * (v3 - (unsigned int)result)));
      if ( *(_BYTE *)(result + 16) <= 3u )
      {
        v5 = *(_QWORD *)(a1 + 256);
        v7 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v5 + 256LL);
        v6 = sub_396EAF0(a1, result);
        result = v7(v5, v6, 14);
      }
      if ( v3 == v4 )
        break;
      ++v3;
      LODWORD(result) = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
  }
  return result;
}
