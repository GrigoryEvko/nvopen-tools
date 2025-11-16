// Function: sub_32EA6D0
// Address: 0x32ea6d0
//
void __fastcall sub_32EA6D0(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-48h] BYREF
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  int v15; // [rsp+18h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v14 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v14, v6, 1);
  v7 = *a1;
  v15 = *(_DWORD *)(a2 + 72);
  v8 = sub_33FAF80(
         v7,
         216,
         (unsigned int)&v14,
         **(unsigned __int16 **)(a2 + 48),
         *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
         0,
         a3);
  sub_34161C0(*a1, a2, 0, v8, v9);
  sub_34161C0(*a1, a2, 1, a3, 1);
  if ( *(_DWORD *)(v8 + 24) != 328 )
  {
    v13 = v8;
    sub_32B3B20((__int64)(a1 + 71), &v13);
    if ( *(int *)(v8 + 88) < 0 )
    {
      *(_DWORD *)(v8 + 88) = *((_DWORD *)a1 + 12);
      v12 = *((unsigned int *)a1 + 12);
      if ( v12 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v12 + 1, 8u, v10, v11);
        v12 = *((unsigned int *)a1 + 12);
      }
      *(_QWORD *)(a1[5] + 8 * v12) = v8;
      ++*((_DWORD *)a1 + 12);
    }
  }
  sub_32CF870((__int64)a1, a2);
  if ( v14 )
    sub_B91220((__int64)&v14, v14);
}
