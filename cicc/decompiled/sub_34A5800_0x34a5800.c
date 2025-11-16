// Function: sub_34A5800
// Address: 0x34a5800
//
__int64 __fastcall sub_34A5800(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  int v10; // edi
  __int64 v11; // r8

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(v7 + 12) < *(_DWORD *)(v7 + 8) )
    {
      v8 = *(_QWORD *)a1;
      v9 = *(unsigned int *)(*(_QWORD *)a1 + 192LL);
      if ( (_DWORD)v9 )
      {
        return sub_34A5670(a1, a2, v7, v8, v9, a6);
      }
      else
      {
        v10 = *(_DWORD *)(v8 + 196);
        v11 = v7 + 16 * result - 16;
        for ( result = *(unsigned int *)(v11 + 12); v10 != (_DWORD)result; result = (unsigned int)(result + 1) )
        {
          if ( a2 <= *(_QWORD *)(v8 + 16LL * (unsigned int)result + 8) )
            break;
        }
        *(_DWORD *)(v11 + 12) = result;
      }
    }
  }
  return result;
}
