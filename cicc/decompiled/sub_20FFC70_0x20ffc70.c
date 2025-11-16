// Function: sub_20FFC70
// Address: 0x20ffc70
//
__int64 __fastcall sub_20FFC70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 (*v9)(); // rax
  __int64 *v10; // rax
  __int64 *v12; // rsi
  unsigned int v13; // edi
  __int64 *v14; // rcx
  char v15; // al
  __int64 v16; // [rsp+8h] [rbp-28h]

  *(_BYTE *)(a1 + 68) = 1;
  v6 = *(_QWORD *)(a3 + 16);
  if ( *(_WORD *)v6 == 9
    || (*(_BYTE *)(v6 + 11) & 2) != 0
    && ((v7 = *(__int64 **)(a1 + 48), v9 = *(__int64 (**)())(*v7 + 16), v9 != sub_1E1C800)
     && (v16 = a4, v15 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64))v9)(v7, a3, a4), a4 = v16, v15)
     || (unsigned __int8)sub_1F3B9C0(v7, a3, a4)) )
  {
    v10 = *(__int64 **)(a1 + 88);
    if ( *(__int64 **)(a1 + 96) != v10 )
      goto LABEL_6;
    v12 = &v10[*(unsigned int *)(a1 + 108)];
    v13 = *(_DWORD *)(a1 + 108);
    if ( v10 != v12 )
    {
      v14 = 0;
      while ( a2 != *v10 )
      {
        if ( *v10 == -2 )
          v14 = v10;
        if ( v12 == ++v10 )
        {
          if ( !v14 )
            goto LABEL_19;
          *v14 = a2;
          --*(_DWORD *)(a1 + 112);
          ++*(_QWORD *)(a1 + 80);
          return 1;
        }
      }
      return 1;
    }
LABEL_19:
    if ( v13 < *(_DWORD *)(a1 + 104) )
    {
      *(_DWORD *)(a1 + 108) = v13 + 1;
      *v12 = a2;
      ++*(_QWORD *)(a1 + 80);
    }
    else
    {
LABEL_6:
      sub_16CCBA0(a1 + 80, a2);
    }
    return 1;
  }
  return 0;
}
