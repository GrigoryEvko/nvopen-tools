// Function: sub_A5EB50
// Address: 0xa5eb50
//
_BYTE *__fastcall sub_A5EB50(__int64 a1, const char *a2, __int64 a3)
{
  __int64 v4; // rbx
  void *v5; // rdx
  unsigned __int8 v6; // al
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r8
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  __int64 v16; // r14
  unsigned int v17; // eax
  _BYTE *result; // rax
  __int64 v19; // [rsp+0h] [rbp-40h] BYREF
  char v20; // [rsp+8h] [rbp-38h]
  char *v21; // [rsp+10h] [rbp-30h]
  __int64 v22; // [rsp+18h] [rbp-28h]

  v4 = (__int64)a2;
  v5 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 0xDu )
  {
    a2 = "!DIStringType(";
    sub_CB6200(a1, "!DIStringType(", 14);
  }
  else
  {
    qmemcpy(v5, "!DIStringType(", 14);
    *(_QWORD *)(a1 + 32) += 14LL;
  }
  v22 = a3;
  v19 = a1;
  v20 = 1;
  v21 = ", ";
  if ( (unsigned __int16)sub_AF18C0(v4) != 18 )
  {
    a2 = (const char *)v4;
    sub_A53560(&v19, v4);
  }
  v6 = *(_BYTE *)(v4 - 16);
  v7 = v4 - 16;
  if ( (v6 & 2) != 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(v4 - 32) + 16LL);
    if ( v8 )
    {
LABEL_7:
      v8 = sub_B91420(v8, a2);
      v10 = v9;
      goto LABEL_8;
    }
  }
  else
  {
    v8 = *(_QWORD *)(v4 - 8LL * ((v6 >> 2) & 0xF));
    if ( v8 )
      goto LABEL_7;
  }
  v10 = 0;
LABEL_8:
  sub_A53660(&v19, "name", 4u, v8, v10, 1);
  v11 = *(_BYTE *)(v4 - 16);
  if ( (v11 & 2) != 0 )
    v12 = *(_QWORD *)(v4 - 32);
  else
    v12 = v7 - 8LL * ((v11 >> 2) & 0xF);
  sub_A5CC00((__int64)&v19, "stringLength", 0xCu, *(_QWORD *)(v12 + 24), 1);
  v13 = *(_BYTE *)(v4 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(v4 - 32);
  else
    v14 = v7 - 8LL * ((v13 >> 2) & 0xF);
  sub_A5CC00((__int64)&v19, "stringLengthExpression", 0x16u, *(_QWORD *)(v14 + 32), 1);
  v15 = *(_BYTE *)(v4 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(v4 - 32);
  else
    v16 = v7 - 8LL * ((v15 >> 2) & 0xF);
  sub_A5CC00((__int64)&v19, "stringLocationExpression", 0x18u, *(_QWORD *)(v16 + 40), 1);
  sub_A539C0((__int64)&v19, "size", 4u, *(_QWORD *)(v4 + 24));
  v17 = sub_AF18D0(v4);
  sub_A537C0((__int64)&v19, "align", 5u, v17, 1);
  sub_A53AC0(&v19, "encoding", 8u, *(_DWORD *)(v4 + 44), sub_E09D50, 1);
  result = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == result )
    return (_BYTE *)sub_CB6200(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 32);
  return result;
}
