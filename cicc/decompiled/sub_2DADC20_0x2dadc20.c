// Function: sub_2DADC20
// Address: 0x2dadc20
//
bool __fastcall sub_2DADC20(_QWORD *a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned __int64 v4; // r15
  unsigned int v6; // r12d
  __int64 v7; // rax
  _DWORD *v8; // rcx
  __int64 v9; // r14
  __int16 v10; // ax
  __int64 v12; // rsi
  __int64 v13; // rdx
  char v15; // [rsp+18h] [rbp-38h] BYREF
  _BYTE v16[52]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = *(_QWORD *)(a1[7] + 16LL * (a4[2] & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == a3 )
    return 0;
  v6 = (*a4 >> 8) & 0xFFF;
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  v8 = a4;
  v9 = v7;
  v10 = *(_WORD *)(a2 + 68);
  switch ( v10 )
  {
    case 9:
      if ( (unsigned int)sub_2EAB0A0(a4) != 2 )
      {
LABEL_12:
        if ( v6 )
          return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v9 + 256LL))(
                   v9,
                   v4,
                   a3,
                   v6) == 0;
        return sub_2FF6970(v9, v4, a3, v8) == 0;
      }
      v13 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 144LL);
      break;
    case 19:
      v13 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * ((unsigned int)sub_2EAB0A0(a4) + 1) + 24);
      break;
    case 8:
      v12 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 104LL);
      if ( (_DWORD)v12 )
      {
        if ( !v6 )
        {
          v6 = v12;
          return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v9 + 256LL))(
                   v9,
                   v4,
                   a3,
                   v6) == 0;
        }
        v6 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _DWORD *))(*(_QWORD *)v9 + 296LL))(v9, v12, v6, a4);
        if ( v6 )
          return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v9 + 256LL))(
                   v9,
                   v4,
                   a3,
                   v6) == 0;
        return sub_2FF6970(v9, v4, a3, v8) == 0;
      }
      goto LABEL_12;
    default:
      if ( v6 )
        return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v9 + 256LL))(
                 v9,
                 v4,
                 a3,
                 v6) == 0;
      return sub_2FF6970(v9, v4, a3, v8) == 0;
  }
  if ( (_DWORD)v13 != 0 && v6 != 0 )
    return sub_2FF69E0(v9, v4, v6, a3, v13, (unsigned int)&v15, (__int64)v16) == 0;
  if ( v6 )
    return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD))(*(_QWORD *)v9 + 256LL))(
             v9,
             v4,
             a3,
             v6) == 0;
  if ( !(_DWORD)v13 )
    return sub_2FF6970(v9, v4, a3, v8) == 0;
  return (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, _QWORD))(*(_QWORD *)v9 + 256LL))(
           v9,
           a3,
           v4,
           (unsigned int)v13) == 0;
}
