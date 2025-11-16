// Function: sub_2DAE370
// Address: 0x2dae370
//
__int64 __fastcall sub_2DAE370(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v8; // r15d
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 result; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // r8
  unsigned __int64 v16; // rax
  int v17; // [rsp+8h] [rbp-38h]

  v8 = sub_2EAB0A0(a5);
  switch ( *(_WORD *)(a2 + 68) )
  {
    case 0:
    case 0x14:
      return a3;
    case 8:
      v9 = a1[1];
      v10 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 104LL);
      if ( (_DWORD)v10 )
        return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 312LL))(v9, v10, a3, a4);
      return a3;
    case 9:
      v14 = a1[1];
      v15 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 144LL);
      if ( (_DWORD)v15 )
      {
        v17 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 144LL);
        result = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v14 + 320LL))(
                   v14,
                   (unsigned int)v15,
                   a3,
                   a4);
        LODWORD(v15) = v17;
      }
      else
      {
        result = a3;
      }
      if ( v8 == 2 )
        return result;
      v16 = *(_QWORD *)(*(_QWORD *)(*a1 + 56LL) + 16LL * (*(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL) & 0x7FFFFFFF))
          & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v16 + 44) )
        return ~*(_QWORD *)(*(_QWORD *)(a1[1] + 272LL) + 16LL * (unsigned int)v15) & a3;
      else
        return *(_QWORD *)(v16 + 24);
    case 0x13:
      v12 = a1[1];
      v13 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (unsigned int)(v8 + 1) + 24);
      if ( (_DWORD)v13 )
        return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v12 + 320LL))(
                 v12,
                 v13,
                 a3,
                 a4);
      return a3;
    default:
      BUG();
  }
}
