// Function: sub_E169B0
// Address: 0xe169b0
//
__int64 __fastcall sub_E169B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  char *v16; // rax
  __int64 v17; // rdi
  int v18; // eax
  __int64 result; // rax
  _BYTE *v20; // rdi

  ++*(_DWORD *)(a2 + 32);
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(char **)a2;
  v7 = v4 + 1;
  if ( v4 + 1 > v5 )
  {
    v8 = v4 + 993;
    v9 = 2 * v5;
    if ( v8 > v9 )
      *(_QWORD *)(a2 + 16) = v8;
    else
      *(_QWORD *)(a2 + 16) = v9;
    v10 = realloc(v6);
    *(_QWORD *)a2 = v10;
    v6 = (char *)v10;
    if ( !v10 )
      goto LABEL_31;
    v4 = *(_QWORD *)(a2 + 8);
    v7 = v4 + 1;
  }
  *(_QWORD *)(a2 + 8) = v7;
  v6[v4] = 40;
  sub_E161C0((_QWORD *)(a1 + 32), (char **)a2);
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v13 = v11 + 1;
  if ( v11 + 1 > v12 )
  {
    v14 = v11 + 993;
    v15 = 2 * v12;
    if ( v14 > v15 )
      *(_QWORD *)(a2 + 16) = v14;
    else
      *(_QWORD *)(a2 + 16) = v15;
    v16 = (char *)realloc(*(void **)a2);
    *(_QWORD *)a2 = v16;
    if ( v16 )
    {
      v11 = *(_QWORD *)(a2 + 8);
      v13 = v11 + 1;
      goto LABEL_12;
    }
LABEL_31:
    abort();
  }
  v16 = *(char **)a2;
LABEL_12:
  *(_QWORD *)(a2 + 8) = v13;
  v16[v11] = 41;
  v17 = *(_QWORD *)(a1 + 16);
  if ( v17 )
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v17 + 40LL))(v17, a2);
  v18 = *(_DWORD *)(a1 + 64);
  if ( (v18 & 1) != 0 )
  {
    sub_E12F20((__int64 *)a2, 6u, " const");
    v18 = *(_DWORD *)(a1 + 64);
  }
  if ( (v18 & 2) != 0 )
  {
    sub_E12F20((__int64 *)a2, 9u, " volatile");
    v18 = *(_DWORD *)(a1 + 64);
  }
  if ( (v18 & 4) == 0 )
  {
    result = *(unsigned __int8 *)(a1 + 68);
    if ( (_BYTE)result != 1 )
      goto LABEL_20;
LABEL_30:
    result = (__int64)sub_E12F20((__int64 *)a2, 2u, " &");
    goto LABEL_22;
  }
  sub_E12F20((__int64 *)a2, 9u, " restrict");
  result = *(unsigned __int8 *)(a1 + 68);
  if ( (_BYTE)result == 1 )
    goto LABEL_30;
LABEL_20:
  if ( (_BYTE)result == 2 )
    result = (__int64)sub_E12F20((__int64 *)a2, 3u, " &&");
LABEL_22:
  v20 = *(_BYTE **)(a1 + 48);
  if ( v20 )
    result = sub_E15BE0(v20, a2);
  if ( *(_QWORD *)(a1 + 56) )
  {
    sub_E12F20((__int64 *)a2, 0xAu, " requires ");
    return sub_E15BE0(*(_BYTE **)(a1 + 56), a2);
  }
  return result;
}
