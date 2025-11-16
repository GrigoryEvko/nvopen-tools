// Function: sub_29F3CB0
// Address: 0x29f3cb0
//
__int64 __fastcall sub_29F3CB0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  char *v4; // rax
  size_t v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r13
  int v9; // eax
  char v10; // al

  if ( *(_QWORD *)(a1 + 48) )
    return *(_QWORD *)(a1 + 48);
  v3 = *(_QWORD *)(a1 + 40);
  v4 = (char *)sub_BD5D20(a1);
  v8 = sub_BAA410(v3, v4, v5);
  v9 = *(_DWORD *)(a2 + 52);
  if ( v9 == 3
    || v9 == 1
    && (v10 = *(_BYTE *)(a1 + 32) & 0xF, v6 = (v10 + 14) & 0xF, (unsigned __int8)v6 > 3u)
    && ((v10 + 7) & 0xFu) > 1 )
  {
    *(_DWORD *)(v8 + 8) = 3;
  }
  sub_B2F990(a1, v8, v6, v7);
  return v8;
}
