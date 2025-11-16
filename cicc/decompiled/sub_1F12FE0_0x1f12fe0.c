// Function: sub_1F12FE0
// Address: 0x1f12fe0
//
__int64 __fastcall sub_1F12FE0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r13d
  bool v4; // cf
  unsigned int v5; // r14d
  int v6; // r8d
  int v7; // r9d

  v1 = *(_QWORD *)(a1 + 240);
  *(_DWORD *)(a1 + 336) = 0;
  result = (unsigned int)(10 * *(_DWORD *)(v1 + 288));
  if ( (_DWORD)result )
  {
    v3 = result - 1;
    while ( 1 )
    {
      result = *(unsigned int *)(a1 + 472);
      if ( !(_DWORD)result )
        break;
      v5 = *(_DWORD *)(*(_QWORD *)(a1 + 464) + 4LL * (unsigned int)result - 4);
      *(_DWORD *)(a1 + 472) = result - 1;
      result = sub_1F12B80(a1, v5);
      if ( (_BYTE)result && (result = *(unsigned int *)(*(_QWORD *)(a1 + 264) + 112LL * v5 + 16), (int)result > 0) )
      {
        result = *(unsigned int *)(a1 + 336);
        if ( (unsigned int)result >= *(_DWORD *)(a1 + 340) )
        {
          sub_16CD150(a1 + 328, (const void *)(a1 + 344), 0, 4, v6, v7);
          result = *(unsigned int *)(a1 + 336);
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 328) + 4 * result) = v5;
        ++*(_DWORD *)(a1 + 336);
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
