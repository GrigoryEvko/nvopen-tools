// Function: sub_262DD10
// Address: 0x262dd10
//
void __fastcall sub_262DD10(__int64 **a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 *a5, int a6)
{
  __int64 v6; // r14
  __int64 *v9; // rax
  __int64 v10; // rax
  int v11; // eax
  int v12; // ecx
  int v13; // edx
  _QWORD *v14; // rsi
  __int64 *v15; // rax
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 *v18; // r12
  __int64 *v19; // r8
  unsigned int v20; // edx
  __int64 *v21; // rdi
  _QWORD *v22; // rcx
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 *v25; // r10
  int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // r13
  __int64 v30; // r15
  __int64 v31; // r14
  __int64 *v32; // r14
  int v33; // r11d
  int v34; // ecx
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v37; // [rsp+18h] [rbp-78h] BYREF
  _QWORD *v38; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v39[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v40; // [rsp+40h] [rbp-50h]
  _QWORD *v41; // [rsp+48h] [rbp-48h]
  _QWORD *v42; // [rsp+50h] [rbp-40h]

  v6 = (__int64)a2;
  if ( dword_4FF2AE8 >= a6 )
    a6 = dword_4FF2AE8;
  *((_WORD *)a1 + 20) = 0;
  a1[1] = a4;
  a1[2] = a5;
  *((_DWORD *)a1 + 6) = a6;
  *a1 = a2;
  *(__int64 **)((char *)a1 + 44) = (__int64 *)0xFFFFFFFFLL;
  a1[7] = (__int64 *)sub_BCB2A0((_QWORD *)*a2);
  a1[8] = (__int64 *)sub_BCB2B0((_QWORD *)**a1);
  a1[9] = (__int64 *)sub_BCE3C0((__int64 *)**a1, 0);
  v9 = (__int64 *)sub_BCB2B0((_QWORD *)**a1);
  a1[10] = sub_BCD420(v9, 0);
  a1[11] = (__int64 *)sub_BCB2D0((_QWORD *)**a1);
  a1[12] = (__int64 *)sub_BCE3C0((__int64 *)**a1, 0);
  a1[13] = (__int64 *)sub_BCB2E0((_QWORD *)**a1);
  v10 = sub_AE4420((__int64)(*a1 + 39), **a1, 0);
  v38 = v39;
  a1[14] = (__int64 *)v10;
  a1[15] = (__int64 *)1;
  a1[16] = 0;
  a1[17] = 0;
  a1[18] = 0;
  *((_DWORD *)a1 + 38) = 0;
  a1[20] = 0;
  a1[21] = 0;
  a1[22] = 0;
  a1[23] = 0;
  a1[25] = 0;
  a1[26] = 0;
  a1[27] = 0;
  *((_DWORD *)a1 + 56) = 0;
  sub_261A960((__int64 *)&v38, (_BYTE *)a2[29], a2[29] + a2[30]);
  v12 = *((_DWORD *)a2 + 69);
  v13 = *((_DWORD *)a2 + 71);
  v40 = (_QWORD *)a2[33];
  v11 = (int)v40;
  v41 = (_QWORD *)a2[34];
  v14 = (_QWORD *)a2[35];
  *((_DWORD *)a1 + 7) = (_DWORD)v40;
  v42 = v14;
  if ( v11 == 1 )
  {
    *((_BYTE *)a1 + 40) = 1;
  }
  else if ( v11 != 36 )
  {
    goto LABEL_5;
  }
  v27 = sub_BC0510(a3, &unk_4F82418, v6);
  if ( *(_QWORD *)(v6 + 32) != v6 + 24 )
  {
    v35 = v6;
    v28 = v6 + 24;
    v29 = *(_QWORD *)(v6 + 32);
    v30 = *(_QWORD *)(v27 + 8);
    do
    {
      v31 = 0;
      if ( v29 )
        v31 = v29 - 56;
      if ( !sub_B2FC80(v31) )
      {
        v32 = (__int64 *)(sub_BC1CD0(v30, &unk_4F89C30, v31) + 8);
        if ( (unsigned __int8)sub_DFE430(v32, 0) )
          *((_BYTE *)a1 + 40) = 1;
        if ( (unsigned __int8)sub_DFE430(v32, 1u) )
          *((_BYTE *)a1 + 41) = 1;
      }
      v29 = *(_QWORD *)(v29 + 8);
    }
    while ( v28 != v29 );
    v6 = v35;
  }
  v13 = HIDWORD(v42);
  v12 = HIDWORD(v41);
LABEL_5:
  *((_DWORD *)a1 + 8) = v12;
  *((_DWORD *)a1 + 9) = v13;
  v15 = (__int64 *)sub_BA8CD0(v6, (__int64)"llvm.global.annotations", 0x17u, 0);
  a1[24] = v15;
  if ( v15 && !sub_B2FC80((__int64)v15) )
  {
    v16 = (__int64 *)*(a1[24] - 4);
    v17 = 32LL * (*((_DWORD *)v16 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)v16 + 7) & 0x40) != 0 )
    {
      v18 = (__int64 *)*(v16 - 1);
      v16 = &v18[(unsigned __int64)v17 / 8];
    }
    else
    {
      v18 = &v16[v17 / 0xFFFFFFFFFFFFFFF8LL];
    }
    if ( v18 != v16 )
    {
      while ( 1 )
      {
        v23 = *v18;
        v24 = *((_DWORD *)a1 + 56);
        v36 = *v18;
        if ( !v24 )
          break;
        v19 = a1[26];
        v20 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v21 = &v19[v20];
        v22 = (_QWORD *)*v21;
        if ( v23 != *v21 )
        {
          v33 = 1;
          v25 = 0;
          while ( v22 != (_QWORD *)-4096LL )
          {
            if ( v25 || v22 != (_QWORD *)-8192LL )
              v21 = v25;
            v20 = (v24 - 1) & (v33 + v20);
            v22 = (_QWORD *)v19[v20];
            if ( (_QWORD *)v23 == v22 )
              goto LABEL_15;
            ++v33;
            v25 = v21;
            v21 = &v19[v20];
          }
          v34 = *((_DWORD *)a1 + 54);
          if ( !v25 )
            v25 = v21;
          a1[25] = (__int64 *)((char *)a1[25] + 1);
          v26 = v34 + 1;
          v37 = v25;
          if ( 4 * v26 < 3 * v24 )
          {
            if ( v24 - *((_DWORD *)a1 + 55) - v26 <= v24 >> 3 )
            {
LABEL_19:
              sub_CE2A30((__int64)(a1 + 25), v24);
              sub_DA5B20((__int64)(a1 + 25), &v36, &v37);
              v23 = v36;
              v25 = v37;
              v26 = *((_DWORD *)a1 + 54) + 1;
            }
            *((_DWORD *)a1 + 54) = v26;
            if ( *v25 != -4096 )
              --*((_DWORD *)a1 + 55);
            *v25 = v23;
            goto LABEL_15;
          }
LABEL_18:
          v24 *= 2;
          goto LABEL_19;
        }
LABEL_15:
        v18 += 4;
        if ( v16 == v18 )
          goto LABEL_7;
      }
      a1[25] = (__int64 *)((char *)a1[25] + 1);
      v37 = 0;
      goto LABEL_18;
    }
  }
LABEL_7:
  if ( v38 != v39 )
    j_j___libc_free_0((unsigned __int64)v38);
}
