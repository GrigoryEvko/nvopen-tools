// Function: sub_3374B90
// Address: 0x3374b90
//
void __fastcall sub_3374B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r12
  int v28; // edx
  int v29; // r14d
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // rsi
  __int64 v34; // r12
  __int64 v35; // rdx
  __int64 v36; // r13
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rdx
  int v41; // r9d
  __int64 v42; // r10
  __int64 v43; // r11
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rax
  int v47; // edx
  __int64 v48; // rbx
  int v49; // r12d
  __int128 v50; // [rsp-30h] [rbp-D0h]
  __int128 v51; // [rsp-20h] [rbp-C0h]
  __int128 v52; // [rsp-20h] [rbp-C0h]
  __int128 v53; // [rsp-10h] [rbp-B0h]
  __int64 v54; // [rsp+0h] [rbp-A0h]
  __int64 v55; // [rsp+8h] [rbp-98h]
  __int128 v56; // [rsp+10h] [rbp-90h]
  __int64 v57; // [rsp+10h] [rbp-90h]
  __int64 v58; // [rsp+18h] [rbp-88h]
  __int64 v59; // [rsp+60h] [rbp-40h] BYREF
  int v60; // [rsp+68h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 960);
  v7 = *(unsigned int *)(*(_QWORD *)(a2 - 32) + 44LL);
  v8 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 8 * v7);
  sub_2E33F80(*(_QWORD *)(v6 + 744), v8, -1, v7, a5, a6);
  v9 = sub_B2E500(**(_QWORD **)(a1 + 960));
  if ( (unsigned int)sub_B2A630(v9) - 7 <= 1 )
  {
    if ( v8 != sub_3374B60(a1, *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL))
      || !*(_DWORD *)(*(_QWORD *)(a1 + 856) + 648LL) )
    {
      v32 = *(_QWORD *)(a1 + 864);
      v33 = v8;
      v34 = sub_33EEAD0(v32, v8);
      v36 = v35;
      v59 = 0;
      v42 = sub_3373A60(a1, v33, v35, v37, v38, v39);
      v43 = v40;
      v44 = *(_QWORD *)a1;
      v60 = *(_DWORD *)(a1 + 848);
      if ( v44 )
      {
        if ( &v59 != (__int64 *)(v44 + 48) )
        {
          v45 = *(_QWORD *)(v44 + 48);
          v59 = v45;
          if ( v45 )
          {
            v57 = v42;
            v58 = v40;
            sub_B96E90((__int64)&v59, v45, 1);
            v42 = v57;
            v43 = v58;
          }
        }
      }
      *((_QWORD *)&v53 + 1) = v36;
      *(_QWORD *)&v53 = v34;
      *((_QWORD *)&v52 + 1) = v43;
      *(_QWORD *)&v52 = v42;
      v46 = sub_3406EB0(v32, 301, (unsigned int)&v59, 1, 0, v41, v52, v53);
      v48 = v46;
      v49 = v47;
      if ( v46 )
      {
        nullsub_1875(v46, v32, 0);
        *(_QWORD *)(v32 + 384) = v48;
        *(_DWORD *)(v32 + 392) = v49;
        sub_33E2B60(v32, 0);
      }
      else
      {
        *(_QWORD *)(v32 + 384) = 0;
        *(_DWORD *)(v32 + 392) = v47;
      }
      if ( v59 )
        sub_B91220((__int64)&v59, v59);
    }
  }
  else
  {
    *(_BYTE *)(v8 + 234) = 1;
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 864) + 40LL) + 578LL) = 1;
    v10 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a2 - 64) - 32LL) - 8LL);
    if ( *(_BYTE *)v10 == 21 )
    {
      v12 = *(_QWORD **)(a1 + 960);
      v31 = *(_QWORD *)(*v12 + 80LL);
      if ( !v31 )
        BUG();
      v11 = v31 - 24;
    }
    else
    {
      v11 = *(_QWORD *)(v10 + 40);
      v12 = *(_QWORD **)(a1 + 960);
    }
    v13 = *(_QWORD *)(a1 + 864);
    v14 = v8;
    *(_QWORD *)&v56 = sub_33EEAD0(v13, *(_QWORD *)(v12[7] + 8LL * *(unsigned int *)(v11 + 44)));
    *((_QWORD *)&v56 + 1) = v15;
    v16 = sub_33EEAD0(*(_QWORD *)(a1 + 864), v8);
    v18 = v17;
    v59 = 0;
    v23 = sub_3373A60(a1, v14, v17, v19, v20, v21);
    v24 = v22;
    v25 = *(_QWORD *)a1;
    v60 = *(_DWORD *)(a1 + 848);
    if ( v25 )
    {
      if ( &v59 != (__int64 *)(v25 + 48) )
      {
        v26 = *(_QWORD *)(v25 + 48);
        v59 = v26;
        if ( v26 )
        {
          v54 = v23;
          v55 = v22;
          sub_B96E90((__int64)&v59, v26, 1);
          v23 = v54;
          v24 = v55;
        }
      }
    }
    *((_QWORD *)&v51 + 1) = v18;
    *(_QWORD *)&v51 = v16;
    *((_QWORD *)&v50 + 1) = v24;
    *(_QWORD *)&v50 = v23;
    v27 = sub_340F900(v13, 311, (unsigned int)&v59, 1, 0, v24, v50, v51, v56);
    v29 = v28;
    if ( v59 )
      sub_B91220((__int64)&v59, v59);
    v30 = *(_QWORD *)(a1 + 864);
    if ( v27 )
    {
      nullsub_1875(v27, *(_QWORD *)(a1 + 864), 0);
      *(_QWORD *)(v30 + 384) = v27;
      *(_DWORD *)(v30 + 392) = v29;
      sub_33E2B60(v30, 0);
    }
    else
    {
      *(_QWORD *)(v30 + 384) = 0;
      *(_DWORD *)(v30 + 392) = v29;
    }
  }
}
