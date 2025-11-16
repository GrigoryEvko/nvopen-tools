// Function: sub_3440BD0
// Address: 0x3440bd0
//
__int64 __fastcall sub_3440BD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        unsigned int a8)
{
  void *(__fastcall *v9)(__int64, __int64, __int64, __int64); // rax
  unsigned int v10; // eax
  unsigned int v11; // eax

  v9 = *(void *(__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 2008LL);
  if ( v9 == sub_3440830 )
  {
    v10 = *(_DWORD *)(a6 + 8);
    if ( v10 > 0x40 )
    {
      memset(*(void **)a6, 0, 8 * (((unsigned __int64)v10 + 63) >> 6));
      v11 = *(_DWORD *)(a6 + 24);
      if ( v11 <= 0x40 )
        goto LABEL_4;
    }
    else
    {
      *(_QWORD *)a6 = 0;
      v11 = *(_DWORD *)(a6 + 24);
      if ( v11 <= 0x40 )
      {
LABEL_4:
        *(_QWORD *)(a6 + 16) = 0;
        return 0;
      }
    }
    memset(*(void **)(a6 + 16), 0, 8 * (((unsigned __int64)v11 + 63) >> 6));
    return 0;
  }
  else
  {
    ((void (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD))v9)(a1, a2, a3, a6, a5, *a7, a8);
    return 0;
  }
}
