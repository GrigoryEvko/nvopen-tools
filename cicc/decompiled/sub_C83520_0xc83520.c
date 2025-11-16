// Function: sub_C83520
// Address: 0xc83520
//
__int64 __fastcall sub_C83520(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  int v4; // eax
  __int64 v5; // rdx
  __int64 v7; // rax
  int v8; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v9; // [rsp+8h] [rbp-18h] BYREF

  v4 = sub_C834C0(a2, &v8, a3, a4);
  if ( v4 )
  {
    sub_C63CA0(&v9, v4, v5);
    v7 = v9;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v7 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_DWORD *)a1 = v8;
  }
  return a1;
}
