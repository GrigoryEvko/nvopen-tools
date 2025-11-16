// Function: sub_E65280
// Address: 0xe65280
//
__int64 __fastcall sub_E65280(__int64 a1, const char **a2)
{
  bool v2; // zf
  unsigned __int8 v3; // al
  const char *v4; // r14
  size_t v5; // r13
  int v6; // eax
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  const char *v12; // [rsp+0h] [rbp-C0h] BYREF
  size_t v13; // [rsp+8h] [rbp-B8h]
  __int64 v14; // [rsp+10h] [rbp-B0h]
  _BYTE v15[168]; // [rsp+18h] [rbp-A8h] BYREF

  v2 = *((_BYTE *)a2 + 33) == 1;
  v12 = v15;
  v13 = 0;
  v14 = 128;
  if ( !v2 )
    goto LABEL_6;
  v3 = *((_BYTE *)a2 + 32);
  if ( v3 == 1 )
  {
    v5 = 0;
    v4 = 0;
    goto LABEL_7;
  }
  if ( (unsigned __int8)(v3 - 3) > 3u )
  {
LABEL_6:
    sub_CA0EC0((__int64)a2, (__int64)&v12);
    v5 = v13;
    v4 = v12;
    goto LABEL_7;
  }
  if ( v3 == 4 )
  {
    v4 = *(const char **)*a2;
    v5 = *((_QWORD *)*a2 + 1);
    goto LABEL_7;
  }
  if ( v3 > 4u )
  {
    if ( (unsigned __int8)(v3 - 5) <= 1u )
    {
      v5 = (size_t)a2[1];
      v4 = *a2;
      goto LABEL_7;
    }
LABEL_21:
    BUG();
  }
  if ( v3 != 3 )
    goto LABEL_21;
  v4 = *a2;
  v5 = 0;
  if ( *a2 )
    v5 = strlen(*a2);
LABEL_7:
  v6 = sub_C92610();
  v7 = sub_C92860((__int64 *)(a1 + 1344), v4, v5, v6);
  if ( v7 == -1 || (v8 = *(_QWORD *)(a1 + 1344), v9 = v8 + 8LL * v7, v9 == v8 + 8LL * *(unsigned int *)(a1 + 1352)) )
    v10 = 0;
  else
    v10 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
  if ( v12 != v15 )
    _libc_free(v12, v4);
  return v10;
}
