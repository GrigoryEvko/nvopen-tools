// Function: sub_FC83A0
// Address: 0xfc83a0
//
__int64 __fastcall sub_FC83A0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rdi
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int8 *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned __int8 *v19; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_BYTE **)a1;
  if ( (*v3 & 4) != 0 )
  {
    v6 = sub_FC80D0((__int64)v3, a2, a2);
    v7 = *(unsigned int *)(a1 + 16);
    v8 = v7 + 1;
    if ( v7 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 20) )
      goto LABEL_3;
LABEL_7:
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v8, 8u, v4, v5);
    v7 = *(unsigned int *)(a1 + 16);
    goto LABEL_3;
  }
  sub_B9C990(&v19, a2);
  v12 = v19;
  v19 = 0;
  v6 = (__int64)sub_B95B00(v12, a2, v13, v14, v15);
  if ( v19 )
    sub_BA65D0((__int64)v19, a2, v16, v17, v18);
  sub_FC80D0(*(_QWORD *)a1, a2, v6);
  v7 = *(unsigned int *)(a1 + 16);
  v8 = v7 + 1;
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
    goto LABEL_7;
LABEL_3:
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v7) = v6;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
  *(_DWORD *)(a1 + 16) = v10;
  return *(_QWORD *)(v9 + 8 * v10 - 8);
}
