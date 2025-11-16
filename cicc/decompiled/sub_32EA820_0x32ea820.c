// Function: sub_32EA820
// Address: 0x32ea820
//
unsigned __int64 __fastcall sub_32EA820(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned int v12; // r14d
  int v13; // eax
  __int64 v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-60h]
  char v23; // [rsp+17h] [rbp-49h] BYREF
  __int64 v24; // [rsp+18h] [rbp-48h] BYREF
  __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  int v26; // [rsp+28h] [rbp-38h]

  v9 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *((_QWORD *)v9 + 1);
  v12 = *v9;
  v25 = v10;
  if ( v10 )
  {
    v22 = a4;
    sub_B96E90((__int64)&v25, v10, 1);
    a4 = v22;
  }
  v13 = *(_DWORD *)(a2 + 72);
  v23 = 0;
  v26 = v13;
  v14 = sub_32EA990(a1, a2, a3, a4, a5, &v23);
  v15 = v14;
  v17 = v16;
  if ( v14 )
  {
    if ( *(_DWORD *)(v14 + 24) == 328 || (v24 = v14, sub_32B3B20((__int64)(a1 + 71), &v24), *(int *)(v15 + 88) >= 0) )
    {
      if ( !v23 )
      {
LABEL_7:
        v15 = sub_34070B0(*a1, v15, v17, &v25, v12, v11);
        goto LABEL_8;
      }
    }
    else
    {
      *(_DWORD *)(v15 + 88) = *((_DWORD *)a1 + 12);
      v21 = *((unsigned int *)a1 + 12);
      if ( v21 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v21 + 1, 8u, v18, v19);
        v21 = *((unsigned int *)a1 + 12);
      }
      *(_QWORD *)(a1[5] + 8 * v21) = v15;
      ++*((_DWORD *)a1 + 12);
      if ( !v23 )
        goto LABEL_7;
    }
    sub_32EA6D0(a1, a2, v15);
    goto LABEL_7;
  }
LABEL_8:
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v15;
}
