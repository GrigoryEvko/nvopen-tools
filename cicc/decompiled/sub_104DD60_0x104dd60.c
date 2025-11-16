// Function: sub_104DD60
// Address: 0x104dd60
//
unsigned __int64 __fastcall sub_104DD60(
        int *a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, char *, __int64),
        __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  int v12; // edx
  __int64 v13; // rcx
  unsigned __int64 result; // rax
  unsigned __int64 v15; // rax

  v6 = a3(a4, "StackLifetimePrinterPass]", 24);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_QWORD *)(a2 + 24);
  v11 = v7;
  if ( v10 - (unsigned __int64)v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v8 = *(_BYTE **)(a2 + 32);
    v10 = *(_QWORD *)(a2 + 24);
LABEL_3:
    if ( v10 > (unsigned __int64)v8 )
      goto LABEL_4;
    goto LABEL_11;
  }
  if ( !v7 )
    goto LABEL_3;
  memcpy(v8, v9, v7);
  v15 = *(_QWORD *)(a2 + 24);
  v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
  *(_QWORD *)(a2 + 32) = v8;
  if ( v15 > (unsigned __int64)v8 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    v12 = *a1;
    v13 = *(_QWORD *)(a2 + 24);
    result = *(_QWORD *)(a2 + 32);
    if ( *a1 )
      goto LABEL_5;
LABEL_12:
    if ( v13 - result <= 2 )
    {
      sub_CB6200(a2, "may", 3u);
      result = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *(_BYTE *)(result + 2) = 121;
      *(_WORD *)result = 24941;
      result = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = result;
    }
    goto LABEL_8;
  }
LABEL_11:
  sub_CB5D20(a2, 60);
  v12 = *a1;
  v13 = *(_QWORD *)(a2 + 24);
  result = *(_QWORD *)(a2 + 32);
  if ( !*a1 )
    goto LABEL_12;
LABEL_5:
  if ( v12 == 1 )
  {
    if ( v13 - result <= 3 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"must", 4u);
      result = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *(_DWORD *)result = 1953723757;
      result = *(_QWORD *)(a2 + 32) + 4LL;
      *(_QWORD *)(a2 + 32) = result;
    }
  }
LABEL_8:
  if ( *(_QWORD *)(a2 + 24) <= result )
    return sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *(_BYTE *)result = 62;
  return result;
}
