// Function: sub_32C25A0
// Address: 0x32c25a0
//
__int64 __fastcall sub_32C25A0(__int128 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r10
  __int64 v7; // r11
  __int64 v11; // r15
  __int64 v12; // rsi
  __int128 *v13; // rdx
  __int128 *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r8
  __int128 *v17; // r12
  __int64 v18; // r9
  __int128 *v19; // r15
  __int64 v20; // r12
  unsigned __int16 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  int v24; // r9d
  __int64 v25; // r12
  __int64 v27; // rax
  __int128 v28; // [rsp-30h] [rbp-90h]
  __int128 v29; // [rsp-30h] [rbp-90h]
  __int128 v30; // [rsp-20h] [rbp-80h]
  __int128 v31; // [rsp-20h] [rbp-80h]
  __int64 v33; // [rsp+0h] [rbp-60h]
  __int64 v35; // [rsp+18h] [rbp-48h] BYREF
  __int64 v36; // [rsp+20h] [rbp-40h] BYREF
  int v37; // [rsp+28h] [rbp-38h]

  v6 = a4;
  v7 = a5;
  v11 = *(_QWORD *)*a1;
  v12 = *(_QWORD *)(v11 + 80);
  v36 = v12;
  if ( v12 )
  {
    sub_B96E90((__int64)&v36, v12, 1);
    v6 = a4;
    v7 = a5;
  }
  v13 = a1[3];
  v14 = a1[1];
  v37 = *(_DWORD *)(v11 + 72);
  *((_QWORD *)&v30 + 1) = v7;
  *(_QWORD *)&v30 = v6;
  *((_QWORD *)&v28 + 1) = a3;
  *(_QWORD *)&v28 = a2;
  v15 = sub_3412970(
          (_DWORD)v14,
          72,
          (unsigned int)&v36,
          *(_QWORD *)(*(_QWORD *)a1[2] + 48LL),
          *(_DWORD *)(*(_QWORD *)a1[2] + 68LL),
          a6,
          v28,
          v30,
          *v13);
  v16 = v15;
  if ( *(_DWORD *)(v15 + 24) != 328 )
  {
    v17 = a1[4];
    v35 = v15;
    v33 = v15;
    sub_32B3B20((__int64)v17 + 568, &v35);
    v16 = v33;
    if ( *(int *)(v33 + 88) < 0 )
    {
      *(_DWORD *)(v33 + 88) = *((_DWORD *)v17 + 12);
      v27 = *((unsigned int *)v17 + 12);
      if ( v27 + 1 > (unsigned __int64)*((unsigned int *)v17 + 13) )
      {
        sub_C8D5F0((__int64)v17 + 40, (char *)v17 + 56, v27 + 1, 8u, v33, v18);
        v27 = *((unsigned int *)v17 + 12);
        v16 = v33;
      }
      *(_QWORD *)(*((_QWORD *)v17 + 5) + 8 * v27) = v16;
      ++*((_DWORD *)v17 + 12);
    }
  }
  v19 = a1[1];
  v20 = v16;
  v21 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a1[5] + 48LL) + 16LL * *((unsigned int *)a1[5] + 2));
  v22 = sub_3400BD0((_DWORD)v19, 0, (unsigned int)&v36, *v21, *((_QWORD *)v21 + 1), 0, 0);
  *((_QWORD *)&v31 + 1) = 1;
  *(_QWORD *)&v31 = v20;
  *((_QWORD *)&v29 + 1) = v23;
  *(_QWORD *)&v29 = v22;
  v25 = sub_3412970(
          (_DWORD)v19,
          72,
          (unsigned int)&v36,
          *(_QWORD *)(*(_QWORD *)*a1 + 48LL),
          *(_DWORD *)(*(_QWORD *)*a1 + 68LL),
          v24,
          *a1[5],
          v29,
          v31);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v25;
}
