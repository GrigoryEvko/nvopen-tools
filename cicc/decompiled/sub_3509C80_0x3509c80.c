// Function: sub_3509C80
// Address: 0x3509c80
//
__int64 __fastcall sub_3509C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax

  *(_BYTE *)(a1 + 68) = 1;
  if ( *(_WORD *)(a3 + 68) == 10 && (*(_DWORD *)(a3 + 40) & 0xFFFFFF) == 1 )
  {
    if ( !*(_BYTE *)(a1 + 108) )
      goto LABEL_13;
LABEL_5:
    v6 = *(_QWORD **)(a1 + 88);
    a4 = *(unsigned int *)(a1 + 100);
    a3 = (__int64)&v6[a4];
    if ( v6 == (_QWORD *)a3 )
    {
LABEL_14:
      if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 96) )
        goto LABEL_13;
      *(_DWORD *)(a1 + 100) = a4 + 1;
      *(_QWORD *)a3 = a2;
      ++*(_QWORD *)(a1 + 80);
    }
    else
    {
      while ( a2 != *v6 )
      {
        if ( (_QWORD *)a3 == ++v6 )
          goto LABEL_14;
      }
    }
    return 1;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(a3 + 16) + 27LL) & 0x20) == 0
    || !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 48) + 56LL))(*(_QWORD *)(a1 + 48), a3) )
  {
    return 0;
  }
  if ( *(_BYTE *)(a1 + 108) )
    goto LABEL_5;
LABEL_13:
  sub_C8CC70(a1 + 80, a2, a3, a4, a5, a6);
  return 1;
}
