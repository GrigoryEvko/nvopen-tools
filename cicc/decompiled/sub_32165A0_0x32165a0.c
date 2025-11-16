// Function: sub_32165A0
// Address: 0x32165a0
//
__int64 __fastcall sub_32165A0(__int64 a1, __int64 a2, __int16 a3)
{
  unsigned int v6; // eax
  __int64 *v7; // rdi
  __int64 v8; // rax
  void (*v9)(); // rax
  __int64 result; // rax
  __int64 *v11; // rbx
  _QWORD *v12; // rsi
  char *v13; // [rsp+0h] [rbp-50h] BYREF
  int v14; // [rsp+10h] [rbp-40h]
  __int16 v15; // [rsp+20h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 488) )
  {
    v6 = *(_DWORD *)(*(_QWORD *)(a2 + 200) + 544LL) - 42;
    if ( v6 <= 1 )
    {
      v7 = *(__int64 **)(a2 + 224);
      v8 = *v7;
      v14 = *(_DWORD *)(a1 + 8);
      v13 = "Size: ";
      v9 = *(void (**)())(v8 + 120);
      v15 = 2307;
      if ( v9 != nullsub_98 )
        ((void (__fastcall *)(__int64 *, char **, __int64))v9)(v7, &v13, 1);
      switch ( a3 )
      {
        case 3:
          goto LABEL_22;
        case 4:
          goto LABEL_21;
        case 8:
        case 30:
          goto LABEL_14;
        case 9:
        case 24:
          goto LABEL_13;
        case 10:
          goto LABEL_20;
        default:
          goto LABEL_23;
      }
    }
    switch ( a3 )
    {
      case 3:
        goto LABEL_22;
      case 4:
        goto LABEL_21;
      case 8:
      case 30:
        goto LABEL_16;
      case 9:
      case 24:
        goto LABEL_13;
      case 10:
        goto LABEL_20;
      default:
        goto LABEL_23;
    }
  }
  switch ( a3 )
  {
    case 3:
LABEL_22:
      sub_31DC9F0(a2, *(_DWORD *)(a1 + 8));
      goto LABEL_14;
    case 4:
LABEL_21:
      sub_31DCA10(a2, *(_DWORD *)(a1 + 8));
      goto LABEL_14;
    case 8:
    case 30:
      goto LABEL_8;
    case 9:
    case 24:
LABEL_13:
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a2 + 424LL))(
        a2,
        *(unsigned int *)(a1 + 8),
        0,
        0);
      goto LABEL_14;
    case 10:
LABEL_20:
      sub_31DC9D0(a2, *(_DWORD *)(a1 + 8));
LABEL_14:
      if ( !*(_BYTE *)(a2 + 488) )
        goto LABEL_8;
      v6 = *(_DWORD *)(*(_QWORD *)(a2 + 200) + 544LL) - 42;
LABEL_16:
      if ( v6 <= 1 )
      {
        v12 = *(_QWORD **)a1;
        if ( *(_QWORD *)a1 )
          v12 = (_QWORD *)(*v12 & 0xFFFFFFFFFFFFFFF8LL);
        return sub_32160B0(a2, (unsigned __int64)v12, 0);
      }
      else
      {
LABEL_8:
        result = *(_QWORD *)a1;
        if ( *(_QWORD *)a1 )
        {
          result = *(_QWORD *)result;
          do
          {
            result &= 0xFFFFFFFFFFFFFFF8LL;
            v11 = (__int64 *)result;
            if ( !result )
              break;
            sub_3215FD0(result + 8, a2);
            result = *v11;
          }
          while ( (*v11 & 4) == 0 );
        }
      }
      return result;
    default:
LABEL_23:
      BUG();
  }
}
