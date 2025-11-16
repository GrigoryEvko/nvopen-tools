// Function: sub_76D560
// Address: 0x76d560
//
__int64 __fastcall sub_76D560(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // rdi
  void (__fastcall *v7)(__int64, __int64); // rax
  __int64 v8; // rdi
  __int64 result; // rax
  void (__fastcall *v10)(__int64, __int64); // rax
  char v11; // al
  __int64 v12; // rbx
  __int64 (__fastcall *v13)(_QWORD, __int64); // rdx
  __int64 (__fastcall *v14)(_QWORD, __int64); // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // rdi
  __int64 v18; // r8
  __int64 (__fastcall *v19)(_QWORD, __int64); // rax

  v6 = *(_QWORD **)(a1 + 144);
  if ( v6 && *(_DWORD *)(a2 + 88) )
    return sub_76CDC0(v6, a2, a3, a4, a5);
  v7 = *(void (__fastcall **)(__int64, __int64))(a2 + 64);
  if ( !v7 || (v8 = *(_QWORD *)(a1 + 128)) == 0 || (v7(v8, a2), result = *(unsigned int *)(a2 + 72), !(_DWORD)result) )
  {
    v10 = *(void (__fastcall **)(__int64, __int64))(a2 + 16);
    if ( !v10 )
      goto LABEL_9;
    v10(a1, a2);
    result = *(unsigned int *)(a2 + 72);
    if ( (_DWORD)result )
      return result;
    if ( *(_DWORD *)(a2 + 76) )
    {
      *(_DWORD *)(a2 + 76) = 0;
LABEL_15:
      result = *(_QWORD *)(a2 + 24);
      if ( result )
        return ((__int64 (__fastcall *)(__int64, __int64))result)(a1, a2);
    }
    else
    {
LABEL_9:
      switch ( *(_BYTE *)(a1 + 173) )
      {
        case 6:
          v13 = *(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 64);
          if ( !v13
            || (unsigned __int8)(*(_BYTE *)(a1 + 176) - 4) > 1u
            || (result = v13(*(_QWORD *)(a1 + 184), a2), !*(_DWORD *)(a2 + 72)) )
          {
            if ( !*(_DWORD *)(a2 + 84) || (unsigned __int8)(*(_BYTE *)(a1 + 176) - 2) > 1u )
              goto LABEL_11;
            goto LABEL_34;
          }
          return result;
        case 9:
          sub_76D400(*(_QWORD *)(a1 + 176), a2, a3, a4, a5);
          goto LABEL_11;
        case 0xA:
          v12 = *(_QWORD *)(a1 + 176);
          if ( !v12 )
            goto LABEL_11;
          break;
        case 0xB:
          sub_76D560(*(_QWORD *)(a1 + 176), a2);
          goto LABEL_11;
        case 0xC:
          if ( !*(_DWORD *)(a2 + 92) )
            goto LABEL_11;
          switch ( *(_BYTE *)(a1 + 176) )
          {
            case 0:
            case 2:
            case 3:
            case 0xD:
              break;
            case 1:
            case 5:
            case 6:
            case 7:
            case 8:
            case 9:
            case 0xA:
              v17 = sub_72ECB0(a1);
              if ( v17 )
                sub_76CDC0(v17, a2, v15, v16, v18);
              break;
            case 4:
            case 0xB:
            case 0xC:
              sub_76D560(*(_QWORD *)(a1 + 184), a2);
              break;
            default:
              sub_721090();
          }
          if ( *(_DWORD *)(a2 + 80) )
            goto LABEL_11;
          if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
            goto LABEL_11;
          v19 = *(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 64);
          if ( !v19 )
            goto LABEL_11;
          result = v19(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), a2);
          if ( *(_DWORD *)(a2 + 72) )
            return result;
          goto LABEL_15;
        case 0xF:
          v11 = *(_BYTE *)(a1 + 176);
          switch ( v11 )
          {
            case 6:
              v14 = *(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 64);
              if ( v14 )
              {
                result = v14(*(_QWORD *)(a1 + 184), a2);
                if ( !*(_DWORD *)(a2 + 72) )
                  goto LABEL_15;
                return result;
              }
              break;
            case 13:
              sub_76CDC0(*(_QWORD **)(a1 + 184), a2, a3, a4, a5);
              break;
            case 2:
LABEL_34:
              sub_76D560(*(_QWORD *)(a1 + 184), a2);
              break;
          }
LABEL_11:
          result = *(_QWORD *)(a2 + 24);
          if ( result && !*(_DWORD *)(a2 + 72) )
            return ((__int64 (__fastcall *)(__int64, __int64))result)(a1, a2);
          return result;
        default:
          goto LABEL_11;
      }
      while ( 1 )
      {
        result = sub_76D560(v12, a2);
        if ( *(_DWORD *)(a2 + 72) )
          break;
        v12 = *(_QWORD *)(v12 + 120);
        if ( !v12 )
          goto LABEL_15;
      }
    }
  }
  return result;
}
