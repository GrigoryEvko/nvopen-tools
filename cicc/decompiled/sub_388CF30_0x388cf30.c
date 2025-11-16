// Function: sub_388CF30
// Address: 0x388cf30
//
__int64 __fastcall sub_388CF30(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rdi
  __int64 result; // rax
  unsigned __int64 v5; // rsi
  const char *v6; // [rsp+0h] [rbp-30h] BYREF
  char v7; // [rsp+10h] [rbp-20h]
  char v8; // [rsp+11h] [rbp-1Fh]

  v3 = a1 + 8;
  switch ( *(_DWORD *)(a1 + 64) )
  {
    case 'D':
      *a2 = 1;
      goto LABEL_3;
    case 'E':
      *a2 = 2;
      goto LABEL_3;
    case 'F':
      *a2 = 4;
      goto LABEL_3;
    case 'G':
      *a2 = 5;
      goto LABEL_3;
    case 'H':
      *a2 = 6;
      goto LABEL_3;
    case 'I':
      *a2 = 7;
LABEL_3:
      *(_DWORD *)(a1 + 64) = sub_3887100(v3);
      result = 0;
      break;
    default:
      v5 = *(_QWORD *)(a1 + 56);
      v8 = 1;
      v6 = "Expected ordering on atomic instruction";
      v7 = 3;
      result = sub_38814C0(v3, v5, (__int64)&v6);
      break;
  }
  return result;
}
