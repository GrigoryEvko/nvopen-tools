// Function: sub_E13330
// Address: 0xe13330
//
__int64 __fastcall sub_E13330(__int64 a1, __int64 *a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rdi
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  void *v14; // rdi
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // r13
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 24) )
    sub_E12F20(a2, 2u, "::");
  v4 = a2[1];
  v5 = a2[2];
  v6 = (char *)*a2;
  if ( v4 + 6 > v5 )
  {
    v7 = v4 + 998;
    v8 = 2 * v5;
    if ( v7 > v8 )
      a2[2] = v7;
    else
      a2[2] = v8;
    v9 = realloc(v6);
    *a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_20;
    v4 = a2[1];
  }
  v10 = &v6[v4];
  *(_DWORD *)v10 = 1701602660;
  *((_WORD *)v10 + 2) = 25972;
  v11 = a2[1] + 6;
  a2[1] = v11;
  if ( *(_BYTE *)(a1 + 25) )
  {
    sub_E12F20(a2, 2u, "[]");
    v11 = a2[1];
  }
  v12 = a2[2];
  v13 = v11 + 1;
  v14 = (void *)*a2;
  if ( v11 + 1 > v12 )
  {
    v15 = v11 + 993;
    v16 = 2 * v12;
    if ( v15 > v16 )
      a2[2] = v15;
    else
      a2[2] = v16;
    v17 = realloc(v14);
    *a2 = v17;
    v14 = (void *)v17;
    if ( v17 )
    {
      v11 = a2[1];
      v13 = v11 + 1;
      goto LABEL_15;
    }
LABEL_20:
    abort();
  }
LABEL_15:
  a2[1] = v13;
  *((_BYTE *)v14 + v11) = 32;
  v18 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v18 + 32LL))(v18, a2);
  result = v18[9] & 0xC0;
  if ( (v18[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v18 + 40LL))(v18, a2);
  return result;
}
