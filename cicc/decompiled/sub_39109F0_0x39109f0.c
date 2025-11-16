// Function: sub_39109F0
// Address: 0x39109f0
//
__int64 __fastcall sub_39109F0(__int64 a1, __int64 a2, const void *a3, __int64 a4, const void *a5, unsigned __int64 a6)
{
  int v7; // r13d
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // r14
  signed __int64 v14; // rbx
  int v15; // r8d
  void *v16; // rdi
  __int64 v17; // r9
  void *v18; // rdi
  __int64 v20; // [rsp+8h] [rbp-38h]
  int v21; // [rsp+8h] [rbp-38h]

  v7 = a6;
  v10 = 0;
  v11 = *(unsigned int *)(a2 + 120);
  if ( (_DWORD)v11 )
    v10 = *(_QWORD *)(*(_QWORD *)(a2 + 112) + 32 * v11 - 32);
  v20 = v10;
  result = sub_22077B0(0x140u);
  v13 = result;
  if ( result )
  {
    v14 = 16 * a4;
    sub_38CF760(result, 12, 0, v20);
    v16 = (void *)(v13 + 240);
    *(_WORD *)(v13 + 48) = 0;
    v17 = v14 >> 4;
    *(_QWORD *)(v13 + 64) = v13 + 80;
    *(_QWORD *)(v13 + 72) = 0x2000000000LL;
    *(_QWORD *)(v13 + 112) = v13 + 128;
    *(_QWORD *)(v13 + 120) = 0x400000000LL;
    *(_QWORD *)(v13 + 56) = 0;
    *(_QWORD *)(v13 + 224) = v13 + 240;
    *(_QWORD *)(v13 + 232) = 0x200000000LL;
    if ( (unsigned __int64)v14 > 0x20 )
    {
      sub_16CD150(v13 + 224, (const void *)(v13 + 240), v14 >> 4, 16, v15, v17);
      v17 = v14 >> 4;
      v16 = (void *)(*(_QWORD *)(v13 + 224) + 16LL * *(unsigned int *)(v13 + 232));
    }
    else if ( !v14 )
    {
      goto LABEL_6;
    }
    v21 = v17;
    memcpy(v16, a3, v14);
    LODWORD(v14) = *(_DWORD *)(v13 + 232);
    LODWORD(v17) = v21;
LABEL_6:
    v18 = (void *)(v13 + 288);
    result = 0x2000000000LL;
    *(_DWORD *)(v13 + 232) = v17 + v14;
    *(_QWORD *)(v13 + 272) = v13 + 288;
    *(_QWORD *)(v13 + 280) = 0x2000000000LL;
    if ( a6 > 0x20 )
    {
      sub_16CD150(v13 + 272, (const void *)(v13 + 288), a6, 1, v15, v13 + 272);
      v18 = (void *)(*(_QWORD *)(v13 + 272) + *(unsigned int *)(v13 + 280));
    }
    else if ( !a6 )
    {
LABEL_8:
      *(_DWORD *)(v13 + 280) = v7;
      return result;
    }
    result = (__int64)memcpy(v18, a5, a6);
    v7 = a6 + *(_DWORD *)(v13 + 280);
    goto LABEL_8;
  }
  return result;
}
