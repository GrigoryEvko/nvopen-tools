// Function: sub_32F7110
// Address: 0x32f7110
//
__int64 __fastcall sub_32F7110(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rcx
  _QWORD *v9; // rax
  unsigned int v10; // edx
  int v11; // r9d
  __int64 v12; // r10
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r8
  __int64 v18; // r12
  __int64 v20; // rax
  __int128 v21; // [rsp-10h] [rbp-A0h]
  __int64 v24; // [rsp+40h] [rbp-50h] BYREF
  int v25; // [rsp+48h] [rbp-48h]
  __int64 v26[8]; // [rsp+50h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v24 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v24, v6, 1);
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)(a2 + 112);
  v25 = *(_DWORD *)(a2 + 72);
  v9 = *(_QWORD **)(a2 + 40);
  if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
    v12 = sub_33F49B0(
            v7,
            a3,
            a4,
            (unsigned int)&v24,
            v9[5],
            v9[6],
            v9[10],
            v9[11],
            *(unsigned __int16 *)(a2 + 96),
            *(_QWORD *)(a2 + 104),
            v8);
  else
    v12 = sub_33F3F90(v7, a3, a4, (unsigned int)&v24, v9[5], v9[6], v9[10], v9[11], v8);
  *((_QWORD *)&v21 + 1) = v10;
  *(_QWORD *)&v21 = v12;
  v13 = sub_3406EB0(*(_QWORD *)a1, 2, (unsigned int)&v24, 1, 0, v11, *(_OWORD *)*(_QWORD *)(a2 + 40), v21);
  v14 = v13;
  v16 = v15;
  if ( *(_DWORD *)(v13 + 24) != 328 )
  {
    v26[0] = v13;
    sub_32B3B20(a1 + 568, v26);
    if ( *(int *)(v14 + 88) < 0 )
    {
      *(_DWORD *)(v14 + 88) = *(_DWORD *)(a1 + 48);
      v20 = *(unsigned int *)(a1 + 48);
      if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v20 + 1, 8u, v17, (__int64)v26);
        v20 = *(unsigned int *)(a1 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v20) = v14;
      ++*(_DWORD *)(a1 + 48);
    }
  }
  v26[0] = v14;
  v26[1] = v16;
  v18 = sub_32EB790(a1, a2, v26, 1, 0);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v18;
}
