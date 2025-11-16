// Function: sub_E13ED0
// Address: 0xe13ed0
//
unsigned __int64 __fastcall sub_E13ED0(_QWORD *a1, __int64 *a2)
{
  _BYTE *v4; // r13
  _BYTE *v5; // r13
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  char *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // rdx
  __int64 v13; // rsi
  unsigned __int64 result; // rax
  void *v15; // rdi
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax

  v4 = (_BYTE *)a1[3];
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v4 + 32LL))(v4);
  if ( (v4[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v4 + 40LL))(v4, a2);
  sub_E12F20(a2, 2u, ".<");
  v5 = (_BYTE *)a1[2];
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v5 + 32LL))(v5, a2);
  if ( (v5[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v5 + 40LL))(v5, a2);
  v6 = a2[1];
  v7 = a2[2];
  v8 = (char *)*a2;
  if ( v6 + 11 > v7 )
  {
    v9 = v6 + 1003;
    v10 = 2 * v7;
    if ( v9 <= v10 )
      a2[2] = v10;
    else
      a2[2] = v9;
    v11 = realloc(v8);
    *a2 = v11;
    v8 = (char *)v11;
    if ( !v11 )
      goto LABEL_23;
    v6 = a2[1];
  }
  qmemcpy(&v8[v6], " at offset ", 11);
  a2[1] += 11;
  if ( a1[4] )
  {
    v12 = (_BYTE *)a1[5];
    if ( *v12 == 110 )
    {
      sub_E12F20(a2, 1u, "-");
      sub_E12F20(a2, a1[4] - 1LL, (const void *)(a1[5] + 1LL));
    }
    else
    {
      sub_E12F20(a2, a1[4], v12);
    }
  }
  else
  {
    sub_E12F20(a2, 1u, "0");
  }
  v13 = a2[1];
  result = a2[2];
  v15 = (void *)*a2;
  if ( v13 + 1 > result )
  {
    v16 = v13 + 993;
    v17 = 2 * result;
    if ( v16 > v17 )
      a2[2] = v16;
    else
      a2[2] = v17;
    result = realloc(v15);
    *a2 = result;
    v15 = (void *)result;
    if ( result )
    {
      v13 = a2[1];
      goto LABEL_18;
    }
LABEL_23:
    abort();
  }
LABEL_18:
  *((_BYTE *)v15 + v13) = 62;
  ++a2[1];
  return result;
}
