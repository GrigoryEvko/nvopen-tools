// Function: sub_325D670
// Address: 0x325d670
//
__int64 __fastcall sub_325D670(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __int64 v10; // r14
  unsigned int v11; // edx
  __int64 v12; // r15
  int v13; // r9d
  __int64 v14; // rax
  unsigned int v15; // edx
  __int64 v16; // r12
  unsigned int v17; // edx
  __int64 v18; // r15
  int v19; // r9d
  __int64 v20; // r14
  unsigned int v21; // edx
  __int64 v22; // r13
  int v23; // r9d
  __int64 v24; // r11
  int *v25; // rax
  unsigned int v26; // edx
  unsigned int v27; // r10d
  __int64 v28; // rdx
  int v29; // ecx
  __int64 v30; // r8
  _DWORD *v31; // rax
  __int128 v33; // [rsp-30h] [rbp-A0h]
  __int128 v34; // [rsp-20h] [rbp-90h]
  __int128 v35; // [rsp-20h] [rbp-90h]
  _QWORD *v36; // [rsp+8h] [rbp-68h]
  __int128 v37; // [rsp+10h] [rbp-60h]
  __int128 v38; // [rsp+20h] [rbp-50h]
  __int128 v40; // [rsp+80h] [rbp+10h]
  __int128 v41; // [rsp+90h] [rbp+20h]

  *(_QWORD *)&v38 = a4;
  *((_QWORD *)&v38 + 1) = a5;
  v37 = (__int128)_mm_loadu_si128((const __m128i *)&a9);
  v36 = *(_QWORD **)a1;
  v10 = sub_33FAF80(
          **(_QWORD **)a1,
          233,
          *(_QWORD *)(a1 + 16),
          **(_DWORD **)(a1 + 24),
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
          *(_QWORD *)a1,
          a8);
  v12 = v11;
  v14 = sub_33FAF80(
          **(_QWORD **)a1,
          233,
          *(_QWORD *)(a1 + 16),
          **(_DWORD **)(a1 + 24),
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
          v13,
          a7);
  *((_QWORD *)&v34 + 1) = v12;
  *(_QWORD *)&v34 = v10;
  *((_QWORD *)&v33 + 1) = v15;
  *(_QWORD *)&v33 = v14;
  v16 = sub_340F900(
          *v36,
          **(_DWORD **)(a1 + 8),
          *(_QWORD *)(a1 + 16),
          **(_DWORD **)(a1 + 24),
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
          (_DWORD)v36,
          v33,
          v34,
          v37);
  v18 = v17;
  v20 = sub_33FAF80(
          **(_QWORD **)a1,
          233,
          *(_QWORD *)(a1 + 16),
          **(_DWORD **)(a1 + 24),
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
          v19,
          v38);
  v22 = v21;
  *((_QWORD *)&v35 + 1) = a3;
  *(_QWORD *)&v35 = a2;
  v24 = sub_33FAF80(
          **(_QWORD **)a1,
          233,
          *(_QWORD *)(a1 + 16),
          **(_DWORD **)(a1 + 24),
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
          v23,
          v35);
  v25 = *(int **)(a1 + 24);
  v27 = v26;
  v28 = *(_QWORD *)(a1 + 16);
  v29 = *v25;
  v30 = *((_QWORD *)v25 + 1);
  *(_QWORD *)&a9 = v16;
  v31 = *(_DWORD **)(a1 + 8);
  *((_QWORD *)&a9 + 1) = v18;
  *(_QWORD *)&v41 = v20;
  *((_QWORD *)&v41 + 1) = v22;
  *(_QWORD *)&v40 = v24;
  *((_QWORD *)&v40 + 1) = v27;
  return sub_340F900(*v36, *v31, v28, v29, v30, (_DWORD)v36, v40, v41, a9);
}
