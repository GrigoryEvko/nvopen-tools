// Function: sub_3700420
// Address: 0x3700420
//
void __fastcall sub_3700420(__int64 a1)
{
  size_t v2; // r12
  int *v3; // r15
  int v4; // eax
  unsigned int v5; // r8d
  _QWORD *v6; // rcx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rax
  unsigned int v14; // r8d
  _QWORD *v15; // rcx
  _QWORD *v16; // r13
  _QWORD *v17; // [rsp+0h] [rbp-120h]
  unsigned int v18; // [rsp+Ch] [rbp-114h]
  unsigned __int16 v19; // [rsp+10h] [rbp-110h] BYREF
  __int64 v20; // [rsp+18h] [rbp-108h]
  int *v21; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v22; // [rsp+48h] [rbp-D8h]
  int v23; // [rsp+50h] [rbp-D0h] BYREF
  char v24; // [rsp+54h] [rbp-CCh]
  __int64 v25; // [rsp+D0h] [rbp-50h]
  __int64 v26; // [rsp+D8h] [rbp-48h]
  __int64 v27; // [rsp+E0h] [rbp-40h]
  __int64 v28; // [rsp+E8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 152);
  v3 = *(int **)(a1 + 144);
  v21 = v3;
  v22 = v2;
  v4 = sub_C92610();
  v5 = sub_C92740(a1 + 120, v3, v2, v4);
  v6 = (_QWORD *)(*(_QWORD *)(a1 + 120) + 8LL * v5);
  v7 = *v6;
  if ( *v6 )
  {
    if ( v7 != -8 )
    {
      v8 = *(_QWORD *)(v7 + 8) + 1LL;
      *(_QWORD *)(v7 + 8) = v8;
      goto LABEL_4;
    }
    --*(_DWORD *)(a1 + 136);
  }
  v17 = v6;
  v18 = v5;
  v13 = sub_C7D670(v2 + 17, 8);
  v14 = v18;
  v15 = v17;
  v16 = (_QWORD *)v13;
  if ( v2 )
  {
    memcpy((void *)(v13 + 16), v3, v2);
    v14 = v18;
    v15 = v17;
  }
  *((_BYTE *)v16 + v2 + 16) = 0;
  *v16 = v2;
  v16[1] = 0;
  *v15 = v16;
  v8 = 0;
  ++*(_DWORD *)(a1 + 132);
  sub_C929D0((__int64 *)(a1 + 120), v14);
LABEL_4:
  v9 = *(_QWORD *)a1;
  v25 = 0;
  v21 = &v23;
  v27 = v9;
  v22 = 0x1000000001LL;
  v26 = 0;
  v28 = 0;
  v23 = 0;
  v24 = 0;
  sub_C6ACB0((__int64)&v21);
  v20 = v8;
  v19 = 3;
  sub_C6B410((__int64)&v21, "observation", 0xBu);
  sub_C6C710((__int64)&v21, &v19, v10);
  sub_C6AE10((__int64)&v21);
  sub_C6BC50(&v19);
  sub_C6AD90((__int64)&v21);
  v11 = *(_QWORD *)a1;
  v12 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
  if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v12 )
  {
    sub_CB6200(v11, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v12 = 10;
    ++*(_QWORD *)(v11 + 32);
  }
  if ( v21 != &v23 )
    _libc_free((unsigned __int64)v21);
}
