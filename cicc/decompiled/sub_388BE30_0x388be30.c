// Function: sub_388BE30
// Address: 0x388be30
//
__int64 __fastcall sub_388BE30(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rdi
  int v4; // eax
  unsigned __int64 v5; // rsi
  const char *v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+10h] [rbp-20h]
  char v9; // [rsp+11h] [rbp-1Fh]

  v3 = a1 + 8;
  v4 = *(_DWORD *)(v3 + 56);
  switch ( v4 )
  {
    case '/':
      *a2 = 3;
      break;
    case '0':
      *a2 = 4;
      break;
    case '.':
      *a2 = 2;
      break;
    default:
      v5 = *(_QWORD *)(a1 + 56);
      v9 = 1;
      v7 = "expected localdynamic, initialexec or localexec";
      v8 = 3;
      return sub_38814C0(v3, v5, (__int64)&v7);
  }
  *(_DWORD *)(a1 + 64) = sub_3887100(v3);
  return 0;
}
