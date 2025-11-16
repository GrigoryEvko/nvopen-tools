// Function: sub_21897A0
// Address: 0x21897a0
//
void *__fastcall sub_21897A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v6; // rbx
  void *result; // rax
  char *v8; // rax
  size_t v9; // rdx
  void *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // r13
  __int64 v17; // rax
  size_t v18; // rdx
  void *v19; // rdi
  char *v20; // rsi
  size_t v21; // [rsp+8h] [rbp-48h]
  size_t v22; // [rsp+8h] [rbp-48h]
  _BYTE v23[64]; // [rsp+10h] [rbp-40h] BYREF

  v4 = a4;
  v6 = *(_QWORD *)(a2 + 16) + 16LL * a3;
  if ( *(_BYTE *)v6 == 1 )
    return (void *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 24LL))(
                     a1,
                     a4,
                     *(unsigned int *)(v6 + 8));
  if ( *(_BYTE *)v6 != 2 )
    return (void *)sub_38CDBE0(*(_QWORD *)(v6 + 8), a4, *(_QWORD *)(a1 + 16), 0);
  v8 = (char *)sub_38D0780(a1, "<imm:", 5);
  v10 = *(void **)(v4 + 24);
  if ( v9 > *(_QWORD *)(v4 + 16) - (_QWORD)v10 )
  {
    v4 = sub_16E7EE0(v4, v8, v9);
  }
  else if ( v9 )
  {
    v21 = v9;
    memcpy(v10, v8, v9);
    *(_QWORD *)(v4 + 24) += v21;
  }
  v11 = *(_QWORD *)(v6 + 8);
  if ( *(_BYTE *)(a1 + 41) )
    sub_38D07D0(v23, a1, v11);
  else
    sub_38D07A0(v23, a1, v11);
  v16 = sub_16E8450(v4, (__int64)v23, v12, v13, v14, v15);
  v17 = sub_38D0780(a1, ">", 1);
  v19 = *(void **)(v16 + 24);
  v20 = (char *)v17;
  result = (void *)(*(_QWORD *)(v16 + 16) - (_QWORD)v19);
  if ( (unsigned __int64)result < v18 )
    return (void *)sub_16E7EE0(v16, v20, v18);
  if ( v18 )
  {
    v22 = v18;
    result = memcpy(v19, v20, v18);
    *(_QWORD *)(v16 + 24) += v22;
  }
  return result;
}
