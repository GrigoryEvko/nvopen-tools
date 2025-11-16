// Function: sub_CB7190
// Address: 0xcb7190
//
__int64 __fastcall sub_CB7190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // rdx
  int v8; // eax
  __int64 v10; // rax
  __int64 v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v6 = sub_C83740(*(unsigned int *)(a2 + 48), a2, a3, a4, a5, a6);
  if ( v6 )
  {
    sub_C63CA0(v11, v6, v7);
    v10 = v11[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v10 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    v8 = *(_BYTE *)(a1 + 8) & 0xFC;
    *(_DWORD *)a1 = *(_DWORD *)(a2 + 48);
    *(_BYTE *)(a1 + 8) = v8 | 2;
  }
  return a1;
}
