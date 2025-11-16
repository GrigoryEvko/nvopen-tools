// Function: sub_120C120
// Address: 0x120c120
//
__int64 __fastcall sub_120C120(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rdi
  int v4; // eax
  unsigned __int64 v5; // rsi
  const char *v7; // [rsp+0h] [rbp-40h] BYREF
  char v8; // [rsp+20h] [rbp-20h]
  char v9; // [rsp+21h] [rbp-1Fh]

  v3 = a1 + 176;
  v4 = *(_DWORD *)(v3 + 64);
  switch ( v4 )
  {
    case '1':
      *a2 = 3;
      break;
    case '2':
      *a2 = 4;
      break;
    case '0':
      *a2 = 2;
      break;
    default:
      v5 = *(_QWORD *)(a1 + 232);
      v7 = "expected localdynamic, initialexec or localexec";
      v9 = 1;
      v8 = 3;
      sub_11FD800(v3, v5, (__int64)&v7, 1);
      return 1;
  }
  *(_DWORD *)(a1 + 240) = sub_1205200(v3);
  return 0;
}
