// Function: sub_1D7ED90
// Address: 0x1d7ed90
//
bool __fastcall sub_1D7ED90(__int64 *a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v9; // r8
  unsigned int v10; // r12d
  __int64 (*v11)(); // rax
  __int16 v12; // ax
  __int64 v14; // rsi
  unsigned int v15; // r8d
  __int64 v16; // rax
  char v17; // [rsp+18h] [rbp-28h] BYREF
  _BYTE v18[36]; // [rsp+1Ch] [rbp-24h] BYREF

  v4 = *(_QWORD *)(a1[3] + 16LL * (a4[2] & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3 == v4 )
    return 0;
  v5 = *a1;
  v6 = 0;
  v9 = *(_QWORD *)(v5 + 16);
  v10 = (*a4 >> 8) & 0xFFF;
  v11 = *(__int64 (**)())(*(_QWORD *)v9 + 112LL);
  if ( v11 != sub_1D00B10 )
    v6 = ((__int64 (__fastcall *)(__int64))v11)(v9);
  v12 = **(_WORD **)(a2 + 16);
  if ( v12 == 8 )
  {
    v16 = *(_QWORD *)(a2 + 32);
    if ( -858993459 * (unsigned int)(((__int64)a4 - v16) >> 3) != 2 )
      goto LABEL_7;
    v15 = *(_DWORD *)(v16 + 144);
  }
  else
  {
    if ( v12 != 14 )
    {
      if ( v12 == 7 )
      {
        v14 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 104LL);
        if ( (_DWORD)v14 )
        {
          if ( v10 )
          {
            v10 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v6 + 120LL))(v6, v14, v10);
            if ( v10 )
              return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v6 + 96LL))(
                       v6,
                       v4,
                       a3,
                       v10) == 0;
            return sub_1F4AF90(v6, v4, a3, 255) == 0;
          }
          v10 = v14;
          return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v6 + 96LL))(
                   v6,
                   v4,
                   a3,
                   v10) == 0;
        }
      }
LABEL_7:
      if ( !v10 )
        return sub_1F4AF90(v6, v4, a3, 255) == 0;
      return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v6 + 96LL))(
               v6,
               v4,
               a3,
               v10) == 0;
    }
    v15 = *(_DWORD *)(*(_QWORD *)(a2 + 32)
                    + 40LL * (-858993459 * (unsigned int)(((__int64)a4 - *(_QWORD *)(a2 + 32)) >> 3) + 1)
                    + 24);
  }
  if ( v10 )
  {
    if ( !v15 )
      return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v6 + 96LL))(
               v6,
               v4,
               a3,
               v10) == 0;
    return sub_1F4B080(v6, v4, v10, a3, v15, (unsigned int)&v17, (__int64)v18) == 0;
  }
  else
  {
    if ( !v15 )
      return sub_1F4AF90(v6, v4, a3, 255) == 0;
    return (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, _QWORD))(*(_QWORD *)v6 + 96LL))(
             v6,
             a3,
             v4,
             v15) == 0;
  }
}
