// Function: sub_120A640
// Address: 0x120a640
//
__int64 __fastcall sub_120A640(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rdi
  int v4; // eax
  unsigned __int64 v6; // rsi
  const char *v7; // [rsp+0h] [rbp-40h] BYREF
  char v8; // [rsp+20h] [rbp-20h]
  char v9; // [rsp+21h] [rbp-1Fh]

  v3 = a1 + 176;
  v4 = *(_DWORD *)(v3 + 64);
  if ( v4 == 25 )
  {
    *a2 = 1;
    goto LABEL_3;
  }
  *a2 = 0;
  if ( v4 == 24 )
  {
LABEL_3:
    *(_DWORD *)(a1 + 240) = sub_1205200(v3);
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 232);
  v7 = "expected 'global' or 'constant'";
  v9 = 1;
  v8 = 3;
  sub_11FD800(v3, v6, (__int64)&v7, 1);
  return 1;
}
