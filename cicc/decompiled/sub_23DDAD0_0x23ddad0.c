// Function: sub_23DDAD0
// Address: 0x23ddad0
//
_QWORD *__fastcall sub_23DDAD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, size_t a5)
{
  __int64 v7; // rsi
  __int64 v9; // r14
  char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r13
  char *v16; // rax
  size_t v17; // rdx
  const char **v18; // r8
  char *v19; // [rsp+8h] [rbp-78h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  _QWORD *v21; // [rsp+20h] [rbp-60h] BYREF
  size_t v22; // [rsp+28h] [rbp-58h]
  _QWORD v23[10]; // [rsp+30h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a2 + 48);
  if ( !v7 )
  {
    v9 = *(_QWORD *)(a2 + 40);
    if ( (*(_BYTE *)(a2 + 7) & 0x10) == 0 )
    {
      v20 = a4;
      sub_23DCD00((__int64)&v21, (__int64)"anon_global", 11);
      sub_BD6B50((unsigned __int8 *)a2, v18);
      a4 = v20;
    }
    if ( a5 && (*(_BYTE *)(a2 + 32) & 0xFu) - 7 <= 1 )
    {
      v19 = (char *)a4;
      v11 = (char *)sub_BD5D20(a2);
      v21 = v23;
      sub_23DC2A0((__int64 *)&v21, v11, (__int64)&v11[v12]);
      if ( a5 > 0x3FFFFFFFFFFFFFFFLL - v22 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)&v21, v19, a5);
      v15 = sub_BAA410(v9, v21, v22);
      if ( v21 != v23 )
        j_j___libc_free_0((unsigned __int64)v21);
      if ( *(_DWORD *)(a1 + 100) != 1 )
        goto LABEL_11;
    }
    else
    {
      v16 = (char *)sub_BD5D20(a2);
      v15 = sub_BAA410(v9, v16, v17);
      if ( *(_DWORD *)(a1 + 100) != 1 )
      {
LABEL_11:
        sub_B2F990(a2, v15, v13, v14);
        v7 = *(_QWORD *)(a2 + 48);
        return sub_B2F990(a3, v7, a3, a4);
      }
    }
    *(_DWORD *)(v15 + 8) = 3;
    if ( (*(_BYTE *)(a2 + 32) & 0xF) == 8 )
      *(_WORD *)(a2 + 32) = *(_WORD *)(a2 + 32) & 0xBCC0 | 0x4007;
    goto LABEL_11;
  }
  return sub_B2F990(a3, v7, a3, a4);
}
