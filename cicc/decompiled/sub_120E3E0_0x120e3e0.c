// Function: sub_120E3E0
// Address: 0x120e3e0
//
__int64 __fastcall sub_120E3E0(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rdi
  __int64 result; // rax
  unsigned __int64 v5; // rsi
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+20h] [rbp-20h]
  char v8; // [rsp+21h] [rbp-1Fh]

  v3 = a1 + 176;
  switch ( *(_DWORD *)(a1 + 240) )
  {
    case 'F':
      *a2 = 1;
      goto LABEL_3;
    case 'G':
      *a2 = 2;
      goto LABEL_3;
    case 'H':
      *a2 = 4;
      goto LABEL_3;
    case 'I':
      *a2 = 5;
      goto LABEL_3;
    case 'J':
      *a2 = 6;
      goto LABEL_3;
    case 'K':
      *a2 = 7;
LABEL_3:
      *(_DWORD *)(a1 + 240) = sub_1205200(v3);
      result = 0;
      break;
    default:
      v5 = *(_QWORD *)(a1 + 232);
      v6 = "Expected ordering on atomic instruction";
      v8 = 1;
      v7 = 3;
      sub_11FD800(v3, v5, (__int64)&v6, 1);
      result = 1;
      break;
  }
  return result;
}
