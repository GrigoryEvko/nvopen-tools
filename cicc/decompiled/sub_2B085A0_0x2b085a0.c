// Function: sub_2B085A0
// Address: 0x2b085a0
//
__int64 __fastcall sub_2B085A0(_QWORD **a1, unsigned int a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rsi
  __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rcx
  unsigned int v7; // edx
  unsigned int v8; // edx

  result = 0;
  if ( *(_BYTE *)(**a1 + 8LL * a2 + 4) )
  {
    v3 = *a1[1] + ((unsigned __int64)a2 << 6);
    v4 = *(__int64 **)v3;
    v5 = *(_QWORD *)(*(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8) - 8);
    result = *(unsigned int *)(v5 + 120);
    if ( (_DWORD)result )
    {
      v6 = *v4;
      v7 = *(_DWORD *)(*v4 + 120);
      if ( v7 )
        goto LABEL_4;
    }
    else
    {
      result = *(unsigned int *)(v5 + 8);
      v6 = *v4;
      v7 = *(_DWORD *)(*v4 + 120);
      if ( v7 )
      {
LABEL_4:
        if ( (unsigned int)result < v7 )
          return v7;
        return result;
      }
    }
    v8 = *(_DWORD *)(v6 + 8);
    if ( (unsigned int)result < v8 )
      return v8;
  }
  return result;
}
