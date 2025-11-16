// Function: sub_140C910
// Address: 0x140c910
//
__int64 __fastcall sub_140C910(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // edx
  unsigned int v5; // edx
  __int64 v6; // [rsp-20h] [rbp-20h]

  result = a1;
  if ( *(_BYTE *)(a2 + 18) || *(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8 )
  {
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
  }
  else
  {
    v4 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
    {
      sub_16A4FD0(a1, a2 + 24);
      result = a1;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)(a2 + 24);
    }
    v5 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(result + 24) = v5;
    if ( v5 > 0x40 )
    {
      v6 = result;
      sub_16A4FD0(result + 16, a2 + 24);
      return v6;
    }
    else
    {
      *(_QWORD *)(result + 16) = *(_QWORD *)(a2 + 24);
    }
  }
  return result;
}
