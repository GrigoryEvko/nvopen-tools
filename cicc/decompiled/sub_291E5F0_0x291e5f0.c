// Function: sub_291E5F0
// Address: 0x291e5f0
//
unsigned __int64 __fastcall sub_291E5F0(
        _BYTE *a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, char *, __int64),
        __int64 a4)
{
  unsigned __int8 *v5; // rax
  size_t v6; // rdx
  __int64 v7; // rbx
  _QWORD *v8; // r13
  char *v9; // r14
  unsigned __int64 result; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  char *v13; // r13
  char *v14; // r14
  unsigned int v15; // r13d
  unsigned int v16; // r13d
  __int64 v17; // rcx
  size_t v18; // [rsp+8h] [rbp-38h]

  v5 = (unsigned __int8 *)a3(a4, "SROAPass]", 8);
  v7 = *(_QWORD *)(a2 + 24);
  v8 = *(_QWORD **)(a2 + 32);
  if ( v7 - (__int64)v8 < v6 )
  {
    sub_CB6200(a2, v5, v6);
    v8 = *(_QWORD **)(a2 + 32);
    v7 = *(_QWORD *)(a2 + 24);
  }
  else if ( v6 )
  {
    v18 = v6;
    memcpy(*(void **)(a2 + 32), v5, v6);
    v7 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 32) += v18;
    v8 = *(_QWORD **)(a2 + 32);
  }
  v9 = "<preserve-cfg>";
  if ( *a1 != 1 )
    v9 = "<modify-cfg>";
  result = strlen(v9);
  v11 = result;
  if ( result > v7 - (__int64)v8 )
    return sub_CB6200(a2, (unsigned __int8 *)v9, result);
  v12 = (unsigned __int64)(v8 + 1) & 0xFFFFFFFFFFFFFFF8LL;
  *v8 = *(_QWORD *)v9;
  result = (unsigned int)result;
  *(_QWORD *)((char *)v8 + (unsigned int)v11 - 8) = *(_QWORD *)&v9[(unsigned int)v11 - 8];
  v13 = (char *)v8 - v12;
  v14 = (char *)(v9 - v13);
  v15 = (v11 + (_DWORD)v13) & 0xFFFFFFF8;
  if ( v15 >= 8 )
  {
    v16 = v15 & 0xFFFFFFF8;
    LODWORD(result) = 0;
    do
    {
      v17 = (unsigned int)result;
      result = (unsigned int)(result + 8);
      *(_QWORD *)(v12 + v17) = *(_QWORD *)&v14[v17];
    }
    while ( (unsigned int)result < v16 );
  }
  *(_QWORD *)(a2 + 32) += v11;
  return result;
}
