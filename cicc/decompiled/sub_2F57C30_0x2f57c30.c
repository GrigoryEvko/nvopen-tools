// Function: sub_2F57C30
// Address: 0x2f57c30
//
__int64 __fastcall sub_2F57C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 (*v10)(); // rax
  __int64 result; // rax
  __int64 v12; // rax
  char v13; // bl
  size_t v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r9
  unsigned __int64 v18; // rax
  __int64 v19; // rbx
  unsigned int v20; // edx
  size_t v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  int v24; // r13d
  int v25; // r14d
  unsigned __int64 v26; // r15
  _DWORD *v27; // rax
  unsigned __int64 v28; // rdx
  unsigned int v29; // [rsp+10h] [rbp-50h]
  size_t v30; // [rsp+18h] [rbp-48h]
  size_t v31; // [rsp+18h] [rbp-48h]
  int v32[2]; // [rsp+28h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(unsigned int *)(a2 + 112);
  v10 = *(__int64 (**)())(*(_QWORD *)v8 + 48LL);
  if ( v10 == sub_2F4C050 )
    goto LABEL_2;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v10)(v8, v9, *(_QWORD *)(a1 + 768)) )
  {
    LODWORD(v9) = *(_DWORD *)(a2 + 112);
LABEL_2:
    if ( *(int *)(*(_QWORD *)(a1 + 920) + 8 * (v9 & 0x7FFFFFFF)) > 3 )
      return 0;
    v12 = sub_2E13500(*(_QWORD *)(a1 + 32), a2);
    v13 = byte_4F826E9[0];
    if ( v12 )
    {
      v30 = strlen("Register Allocation");
      v14 = strlen("regalloc");
      sub_CA08F0(
        (__int64 *)v32,
        "local_split",
        0xBu,
        (__int64)"Local Splitting",
        15,
        v13,
        "regalloc",
        v14,
        "Register Allocation",
        v30);
      sub_2FB1E90(*(_QWORD *)(a1 + 992), a2);
      result = sub_2F560A0(a1, a2, a3, a4);
      if ( !(_DWORD)result && !*(_DWORD *)(a4 + 8) )
        result = sub_2F542E0(a1, a2, a3, a4, v15, v16);
      goto LABEL_9;
    }
    v31 = strlen("Register Allocation");
    v21 = strlen("regalloc");
    sub_CA08F0(
      (__int64 *)v32,
      "global_split",
      0xCu,
      (__int64)"Global Splitting",
      16,
      v13,
      "regalloc",
      v21,
      "Register Allocation",
      v31);
    sub_2FB1E90(*(_QWORD *)(a1 + 992), a2);
    if ( *(int *)(*(_QWORD *)(a1 + 920) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) > 2 )
      goto LABEL_29;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 648LL))(
           *(_QWORD *)(a1 + 8),
           *(_QWORD *)(a1 + 768),
           a2) )
    {
      result = sub_2F57B40((_QWORD *)a1, a2, a3, a4);
      if ( (_DWORD)result )
        goto LABEL_9;
    }
    if ( *(_DWORD *)(a4 + 8) )
      result = 0;
    else
LABEL_29:
      result = sub_2F53AC0(a1, a2, a3, a4, v22, v23);
LABEL_9:
    if ( *(_QWORD *)v32 )
    {
      v29 = result;
      sub_C9E2A0(*(__int64 *)v32);
      return v29;
    }
    return result;
  }
  v18 = *(unsigned int *)(a1 + 928);
  v19 = *(_DWORD *)(a2 + 112) & 0x7FFFFFFF;
  v20 = v19 + 1;
  if ( (int)v19 + 1 > (unsigned int)v18 && v20 != v18 )
  {
    if ( v20 >= v18 )
    {
      v24 = *(_DWORD *)(a1 + 936);
      v25 = *(_DWORD *)(a1 + 940);
      v26 = v20 - v18;
      if ( v20 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
      {
        sub_C8D5F0(a1 + 920, (const void *)(a1 + 936), v20, 8u, v20, v17);
        v18 = *(unsigned int *)(a1 + 928);
      }
      v27 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v18);
      v28 = v26;
      do
      {
        if ( v27 )
        {
          *v27 = v24;
          v27[1] = v25;
        }
        v27 += 2;
        --v28;
      }
      while ( v28 );
      *(_DWORD *)(a1 + 928) += v26;
    }
    else
    {
      *(_DWORD *)(a1 + 928) = v20;
    }
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v19) = 4;
  return 0;
}
