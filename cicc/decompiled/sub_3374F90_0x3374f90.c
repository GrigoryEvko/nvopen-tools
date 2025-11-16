// Function: sub_3374F90
// Address: 0x3374f90
//
void __fastcall sub_3374F90(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdi
  __int64 v4; // r12
  unsigned int v5; // eax
  unsigned int v6; // edx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rbx
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // r10
  __int64 v22; // r11
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rbx
  int v26; // edx
  int v27; // r12d
  __int64 v28; // r13
  __int128 v29; // [rsp-20h] [rbp-C0h]
  __int128 v30; // [rsp-10h] [rbp-B0h]
  __int64 v31; // [rsp+0h] [rbp-A0h]
  __int64 v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+40h] [rbp-60h] BYREF
  int v34; // [rsp+48h] [rbp-58h]
  __int64 *v35; // [rsp+50h] [rbp-50h] BYREF
  __int64 v36; // [rsp+58h] [rbp-48h]
  _BYTE v37[64]; // [rsp+60h] [rbp-40h] BYREF

  v35 = (__int64 *)v37;
  v3 = *(__int64 **)(a1 + 960);
  v36 = 0x100000000LL;
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    v4 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    if ( v3[4] && v4 )
    {
      v5 = sub_FF0430(v3[4], *(_QWORD *)(v3[93] + 16), v4);
      v3 = *(__int64 **)(a1 + 960);
      v6 = v5;
      goto LABEL_7;
    }
  }
  else
  {
    v4 = 0;
  }
  v6 = 0;
LABEL_7:
  sub_33669C0(v3, v4, v6, (__int64)&v35);
  v9 = v35;
  v10 = &v35[2 * (unsigned int)v36];
  if ( v10 != v35 )
  {
    do
    {
      v11 = *v9;
      v9 += 2;
      *(_BYTE *)(v11 + 216) = 1;
      sub_3373E10(a1, *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL), *(v9 - 2), *((unsigned int *)v9 - 2), v7, v8);
    }
    while ( v10 != v9 );
  }
  sub_2E33470(
    *(unsigned int **)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL) + 144LL),
    *(unsigned int **)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL) + 152LL));
  v12 = *(_QWORD *)(a1 + 864);
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 56LL)
                  + 8LL
                  * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 40LL)
                                    + 44LL));
  v14 = sub_33EEAD0(v12, v13);
  v16 = v15;
  v33 = 0;
  v21 = sub_3373A60(a1, v13, v15, v17, v18, v19);
  v22 = v20;
  v23 = *(_QWORD *)a1;
  v34 = *(_DWORD *)(a1 + 848);
  if ( v23 )
  {
    if ( &v33 != (__int64 *)(v23 + 48) )
    {
      v24 = *(_QWORD *)(v23 + 48);
      v33 = v24;
      if ( v24 )
      {
        v31 = v21;
        v32 = v20;
        sub_B96E90((__int64)&v33, v24, 1);
        v21 = v31;
        v22 = v32;
      }
    }
  }
  *((_QWORD *)&v30 + 1) = v16;
  *(_QWORD *)&v30 = v14;
  *((_QWORD *)&v29 + 1) = v22;
  *(_QWORD *)&v29 = v21;
  v25 = sub_3406EB0(v12, 312, (unsigned int)&v33, 1, 0, (unsigned int)&v33, v29, v30);
  v27 = v26;
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  v28 = *(_QWORD *)(a1 + 864);
  if ( v25 )
  {
    nullsub_1875(v25, *(_QWORD *)(a1 + 864), 0);
    *(_QWORD *)(v28 + 384) = v25;
    *(_DWORD *)(v28 + 392) = v27;
    sub_33E2B60(v28, 0);
  }
  else
  {
    *(_QWORD *)(v28 + 384) = 0;
    *(_DWORD *)(v28 + 392) = v27;
  }
  if ( v35 != (__int64 *)v37 )
    _libc_free((unsigned __int64)v35);
}
