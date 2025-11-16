// Function: sub_E13160
// Address: 0xe13160
//
__int64 __fastcall sub_E13160(_QWORD *a1, __int64 *a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  void *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // r13
  _BYTE *v12; // r13
  __int64 v13; // rsi
  unsigned __int64 v14; // rax
  void *v15; // rdi
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // r13
  __int64 result; // rax

  v4 = a2[1];
  v5 = a2[2];
  v6 = (void *)*a2;
  v7 = v4 + 1;
  if ( v4 + 1 > v5 )
  {
    v8 = v4 + 993;
    v9 = 2 * v5;
    if ( v8 > v9 )
      a2[2] = v8;
    else
      a2[2] = v9;
    v10 = realloc(v6);
    *a2 = v10;
    v6 = (void *)v10;
    if ( !v10 )
      goto LABEL_22;
    v4 = a2[1];
    v7 = v4 + 1;
  }
  a2[1] = v7;
  *((_BYTE *)v6 + v4) = 91;
  v11 = (_BYTE *)a1[2];
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v11 + 32LL))(v11, a2);
  if ( (v11[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v11 + 40LL))(v11, a2);
  sub_E12F20(a2, 5u, " ... ");
  v12 = (_BYTE *)a1[3];
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v12 + 32LL))(v12, a2);
  if ( (v12[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v12 + 40LL))(v12, a2);
  v13 = a2[1];
  v14 = a2[2];
  v15 = (void *)*a2;
  v16 = v13 + 1;
  if ( v13 + 1 > v14 )
  {
    v17 = v13 + 993;
    v18 = 2 * v14;
    if ( v17 > v18 )
      a2[2] = v17;
    else
      a2[2] = v18;
    v19 = realloc(v15);
    *a2 = v19;
    v15 = (void *)v19;
    if ( v19 )
    {
      v13 = a2[1];
      v16 = v13 + 1;
      goto LABEL_15;
    }
LABEL_22:
    abort();
  }
LABEL_15:
  a2[1] = v16;
  *((_BYTE *)v15 + v13) = 93;
  v20 = (_BYTE *)a1[4];
  if ( (unsigned __int8)(v20[8] - 81) > 1u )
  {
    sub_E12F20(a2, 3u, " = ");
    v20 = (_BYTE *)a1[4];
  }
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v20 + 32LL))(v20, a2);
  result = v20[9] & 0xC0;
  if ( (v20[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v20 + 40LL))(v20, a2);
  return result;
}
