// Function: sub_1ED87E0
// Address: 0x1ed87e0
//
__int64 __fastcall sub_1ED87E0(__int64 a1, __int64 a2, _DWORD *a3, _DWORD *a4, int *a5, int *a6)
{
  __int16 v9; // ax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // esi
  unsigned int v14; // eax
  unsigned int v15; // eax
  int *v16; // [rsp+8h] [rbp-28h]

  v9 = **(_WORD **)(a2 + 16);
  if ( v9 == 15 )
  {
    *a4 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    *a6 = (**(_DWORD **)(a2 + 32) >> 8) & 0xFFF;
    *a3 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 48LL);
    *a5 = (*(_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL) >> 8) & 0xFFF;
    return 1;
  }
  else if ( v9 == 10 )
  {
    *a4 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    v11 = *(_QWORD *)(a2 + 32);
    v12 = *(_QWORD *)(v11 + 144);
    v13 = v12;
    v14 = (*(_DWORD *)v11 >> 8) & 0xFFF;
    if ( v14 )
    {
      v13 = v14;
      if ( (_DWORD)v12 )
      {
        v16 = a6;
        v15 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _DWORD *, _QWORD))(*(_QWORD *)a1 + 120LL))(
                a1,
                v14,
                v12,
                a4,
                0);
        a6 = v16;
        v13 = v15;
      }
    }
    *a6 = v13;
    *a3 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 88LL);
    *a5 = (*(_DWORD *)(*(_QWORD *)(a2 + 32) + 80LL) >> 8) & 0xFFF;
    return 1;
  }
  else
  {
    return 0;
  }
}
