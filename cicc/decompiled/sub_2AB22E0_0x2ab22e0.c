// Function: sub_2AB22E0
// Address: 0x2ab22e0
//
__int64 __fastcall sub_2AB22E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  __int64 *v8; // r12
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rbx
  __int64 *v12; // r13
  unsigned __int64 v13; // rbx
  bool v14; // dl
  int v15; // eax
  __int64 *v16; // rax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 *v20; // rsi
  __int64 v21; // rdx
  void *v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  void *v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  void *v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  int v34; // r9d
  int v35; // r10d
  __int64 v36; // [rsp+8h] [rbp-D8h]
  __int64 v37; // [rsp+10h] [rbp-D0h]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  unsigned __int8 v39; // [rsp+27h] [rbp-B9h]
  __int64 v40; // [rsp+28h] [rbp-B8h]
  __int64 v41; // [rsp+30h] [rbp-B0h] BYREF
  bool v42; // [rsp+38h] [rbp-A8h]
  __int64 v43; // [rsp+40h] [rbp-A0h] BYREF
  char v44; // [rsp+48h] [rbp-98h]
  _QWORD v45[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v46; // [rsp+60h] [rbp-80h]
  _QWORD v47[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 *v48; // [rsp+80h] [rbp-60h]
  _QWORD v49[10]; // [rsp+90h] [rbp-50h] BYREF

  v37 = a1 + 56;
  v7 = *(_QWORD *)(a1 + 8) == 0;
  v41 = a1 + 56;
  v36 = a1 + 928;
  v43 = a1 + 928;
  v44 = 0;
  v42 = v7;
  if ( !*(_QWORD *)(a1 + 24) )
  {
    v44 = 1;
    goto LABEL_27;
  }
  v38 = *(_QWORD *)(a1 + 928);
  a2 = *(_QWORD *)(a1 + 16) + 48LL;
  v40 = a2;
  v8 = (__int64 *)(*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (__int64 *)a2 != v8 )
  {
    while ( 1 )
    {
      v11 = *v8;
      a5 = (__int64)v8;
      v47[0] = 0;
      v12 = v8 - 3;
      v47[1] = 0;
      v13 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      v48 = v8 - 3;
      v14 = v8 + 1021 != 0;
      v7 = v8 + 509 == 0;
      v8 = (__int64 *)v13;
      LOBYTE(a3) = !v7 && v14;
      if ( (_BYTE)a3 )
      {
        v39 = a3;
        sub_BD73F0((__int64)v47);
        a3 = v39;
      }
      v15 = *(_DWORD *)(a1 + 1016);
      if ( v15 )
      {
        a4 = v48;
        a2 = (unsigned int)(v15 - 1);
        v9 = *(_QWORD *)(a1 + 1000);
        v10 = a2 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        a6 = *(_QWORD *)(v9 + 24LL * v10 + 16);
        if ( v48 == (__int64 *)a6 )
        {
LABEL_5:
          if ( v48 && v48 != (__int64 *)-4096LL && v48 != (__int64 *)-8192LL )
            sub_BD60C0(v47);
          goto LABEL_9;
        }
        v35 = 1;
        while ( a6 != -4096 )
        {
          a5 = (unsigned int)(v35 + 1);
          v10 = a2 & (v35 + v10);
          a6 = *(_QWORD *)(v9 + 24LL * v10 + 16);
          if ( v48 == (__int64 *)a6 )
            goto LABEL_5;
          ++v35;
        }
      }
      v45[0] = 0;
      v16 = v12;
      v45[1] = 0;
      v46 = v12;
      if ( (_BYTE)a3 )
      {
        sub_BD73F0((__int64)v45);
        v16 = v46;
      }
      a3 = *(unsigned int *)(a1 + 1048);
      if ( (_DWORD)a3 )
      {
        v17 = a3 - 1;
        v18 = *(_QWORD *)(a1 + 1032);
        v49[2] = -4096;
        v49[0] = 0;
        v19 = (a3 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v49[1] = 0;
        v20 = *(__int64 **)(v18 + 24LL * v19 + 16);
        if ( v20 == v16 )
        {
LABEL_17:
          sub_D68D70(v49);
          v16 = v46;
          a2 = 1;
          goto LABEL_18;
        }
        v34 = 1;
        while ( v20 != (__int64 *)-4096LL )
        {
          v19 = v17 & (v34 + v19);
          v20 = *(__int64 **)(v18 + 24LL * v19 + 16);
          if ( v16 == v20 )
            goto LABEL_17;
          ++v34;
        }
        sub_D68D70(v49);
        v16 = v46;
      }
      a2 = 0;
LABEL_18:
      LOBYTE(a4) = v16 != 0;
      LOBYTE(a3) = v16 + 512 != 0;
      if ( ((unsigned __int8)a3 & (v16 != 0)) != 0 && v16 != (__int64 *)-8192LL )
      {
        sub_BD60C0(v45);
        a2 = (unsigned __int8)a2;
      }
      if ( v48 && v48 != (__int64 *)-8192LL && v48 != (__int64 *)-4096LL )
      {
        sub_BD60C0(v47);
        a2 = (unsigned __int8)a2;
      }
      if ( (_BYTE)a2 )
      {
LABEL_9:
        if ( v40 == v13 )
          break;
      }
      else
      {
        a2 = (__int64)v12;
        sub_DAC8D0(v38, v12);
        sub_B43D60(v12);
        if ( v40 == v13 )
          break;
      }
    }
  }
LABEL_27:
  sub_F82D10((__int64)&v43, a2, a3, a4, a5, a6);
  sub_F82D10((__int64)&v41, a2, v21, v22, v23, v24);
  if ( *(_QWORD *)(a1 + 8) )
    sub_AA5450(*(_QWORD **)a1);
  if ( *(_QWORD *)(a1 + 24) )
    sub_AA5450(*(_QWORD **)(a1 + 16));
  sub_F82D10((__int64)&v43, a2, v25, v26, v27, v28);
  sub_F82D10((__int64)&v41, a2, v29, v30, v31, v32);
  sub_27C20B0(v36);
  return sub_27C20B0(v37);
}
