// Function: sub_76D400
// Address: 0x76d400
//
__int64 (__fastcall *__fastcall sub_76D400(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5))(__int64, __int64)
{
  __int64 (*v6)(void); // rax
  __int64 (__fastcall *result)(__int64, __int64); // rax
  __int64 v8; // rdi
  _QWORD *v9; // rdi

  v6 = *(__int64 (**)(void))(a2 + 32);
  if ( !v6 )
    goto LABEL_4;
  result = (__int64 (__fastcall *)(__int64, __int64))v6();
  if ( !*(_DWORD *)(a2 + 72) )
  {
    a5 = *(unsigned int *)(a2 + 76);
    if ( !(_DWORD)a5 )
    {
LABEL_4:
      switch ( *(_BYTE *)(a1 + 48) )
      {
        case 0:
        case 1:
          goto LABEL_8;
        case 2:
          goto LABEL_14;
        case 3:
        case 4:
          sub_76CDC0(*(_QWORD **)(a1 + 56), a2, a3, a4, a5);
          goto LABEL_8;
        case 5:
          sub_76D3C0(*(_QWORD **)(a1 + 64), a2, a3, a4, a5);
          goto LABEL_8;
        case 6:
          sub_76D560(*(_QWORD *)(a1 + 56), a2);
          goto LABEL_8;
        case 7:
          v9 = *(_QWORD **)(a1 + 56);
          if ( v9 )
            sub_76CDC0(v9, a2, a3, a4, a5);
          goto LABEL_8;
        case 8:
          if ( (*(_BYTE *)(a1 + 72) & 1) != 0 )
            goto LABEL_6;
LABEL_14:
          if ( *(_DWORD *)(a2 + 84) )
          {
LABEL_6:
            v8 = *(_QWORD *)(a1 + 56);
LABEL_7:
            sub_76D560(v8, a2);
            goto LABEL_8;
          }
          if ( *(_DWORD *)(a2 + 92) )
          {
            v8 = *(_QWORD *)(a1 + 56);
            if ( *(_BYTE *)(v8 + 173) == 12 )
              goto LABEL_7;
          }
LABEL_8:
          result = *(__int64 (__fastcall **)(__int64, __int64))(a2 + 40);
          if ( !result || *(_DWORD *)(a2 + 72) )
            return result;
          return (__int64 (__fastcall *)(__int64, __int64))result(a1, a2);
        case 9:
          if ( !*(_DWORD *)(a2 + 84) )
            goto LABEL_8;
          v8 = *(_QWORD *)(a1 + 56);
          if ( !v8 )
            goto LABEL_8;
          goto LABEL_7;
        default:
          sub_721090();
      }
    }
    *(_DWORD *)(a2 + 76) = 0;
    result = *(__int64 (__fastcall **)(__int64, __int64))(a2 + 40);
    if ( result )
      return (__int64 (__fastcall *)(__int64, __int64))result(a1, a2);
  }
  return result;
}
