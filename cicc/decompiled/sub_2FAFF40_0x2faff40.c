// Function: sub_2FAFF40
// Address: 0x2faff40
//
__int64 __fastcall sub_2FAFF40(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r13d
  bool v4; // cf
  unsigned int v5; // r14d
  __int64 v6; // r8
  __int64 v7; // r9

  v1 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 96) = 0;
  result = (unsigned int)(10 * *(_DWORD *)(v1 + 56));
  if ( (_DWORD)result )
  {
    v3 = result - 1;
    while ( 1 )
    {
      result = *(unsigned int *)(a1 + 232);
      if ( !(_DWORD)result )
        break;
      v5 = *(_DWORD *)(*(_QWORD *)(a1 + 224) + 4LL * (unsigned int)result - 4);
      *(_DWORD *)(a1 + 232) = result - 1;
      result = sub_2FAFB50(a1, v5);
      if ( (_BYTE)result && (result = *(unsigned int *)(*(_QWORD *)(a1 + 24) + 112LL * v5 + 16), (int)result > 0) )
      {
        result = *(unsigned int *)(a1 + 96);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
        {
          sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), result + 1, 4u, v6, v7);
          result = *(unsigned int *)(a1 + 96);
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 88) + 4 * result) = v5;
        ++*(_DWORD *)(a1 + 96);
        v4 = v3-- == 0;
        if ( v4 )
          return result;
      }
      else
      {
        v4 = v3-- == 0;
        if ( v4 )
          return result;
      }
    }
  }
  return result;
}
