// Function: sub_388AD40
// Address: 0x388ad40
//
__int64 __fastcall sub_388AD40(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rdi
  int v4; // eax
  unsigned __int64 v6; // rsi
  const char *v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+10h] [rbp-20h]
  char v9; // [rsp+11h] [rbp-1Fh]

  v3 = a1 + 8;
  v4 = *(_DWORD *)(v3 + 56);
  if ( v4 == 23 )
  {
    *a2 = 1;
    goto LABEL_3;
  }
  *a2 = 0;
  if ( v4 == 22 )
  {
LABEL_3:
    *(_DWORD *)(a1 + 64) = sub_3887100(v3);
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 56);
  v9 = 1;
  v7 = "expected 'global' or 'constant'";
  v8 = 3;
  return sub_38814C0(v3, v6, (__int64)&v7);
}
