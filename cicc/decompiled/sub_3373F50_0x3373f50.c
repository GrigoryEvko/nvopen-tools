// Function: sub_3373F50
// Address: 0x3373f50
//
__int64 __fastcall sub_3373F50(__int64 a1, unsigned int *a2)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 (__fastcall *v6)(__int64, __int64, __int64, __int64); // r14
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // rax
  __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 v17; // rax
  int v18; // edx
  int v19; // eax
  int v20; // edx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // r14
  __int128 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // edx
  __int64 v29; // r13
  __int64 v30; // r12
  unsigned int v31; // r14d
  __int128 v33; // [rsp-40h] [rbp-E0h]
  __int128 v34; // [rsp-20h] [rbp-C0h]
  __int128 v35; // [rsp-10h] [rbp-B0h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  int v37; // [rsp+10h] [rbp-90h]
  int v38; // [rsp+18h] [rbp-88h]
  int v39; // [rsp+20h] [rbp-80h]
  __int64 v40; // [rsp+20h] [rbp-80h]
  unsigned int v41; // [rsp+28h] [rbp-78h]
  unsigned __int16 v42; // [rsp+2Eh] [rbp-72h]
  __int64 v43; // [rsp+50h] [rbp-50h] BYREF
  int v44; // [rsp+58h] [rbp-48h]
  __int64 v45; // [rsp+60h] [rbp-40h]
  __int64 v46; // [rsp+68h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 864);
  v5 = *(_QWORD *)(v4 + 16);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v5 + 1920LL);
  v7 = sub_2E79000(*(__int64 **)(v4 + 40));
  v8 = v7;
  if ( v6 == sub_3366DC0 )
  {
    v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v5 + 32LL);
    if ( v9 == sub_2D42F30 )
    {
      v10 = 0;
      v11 = sub_AE2980(v8, 0)[1];
      v42 = 2;
      if ( v11 != 1 )
      {
        v15 = 3;
        v42 = 3;
        if ( v11 != 2 )
        {
          v14 = 4;
          v42 = 4;
          if ( v11 != 4 )
          {
            v42 = 5;
            if ( v11 != 8 )
            {
              v10 = 6;
              v42 = 6;
              if ( v11 != 16 )
              {
                v13 = 7;
                v42 = 7;
                if ( v11 != 32 )
                {
                  v12 = 8;
                  v42 = 8;
                  if ( v11 != 64 )
                    v42 = 9 * (v11 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v10 = v8;
      v42 = v9(v5, v8, 0);
    }
  }
  else
  {
    v10 = v7;
    v42 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, v7);
  }
  v16 = *(_QWORD *)(a1 + 864);
  v17 = sub_3373A60(a1, v10, v12, v13, v14, v15);
  v39 = v18;
  v41 = *a2;
  v36 = v17;
  v19 = sub_33E5110(v16, v42, 0, 1, 0);
  v38 = v20;
  v44 = v39;
  v43 = v36;
  v37 = v19;
  v45 = sub_33F0B60(v16, v41, v42, 0);
  v46 = v21;
  *((_QWORD *)&v35 + 1) = 2;
  *(_QWORD *)&v35 = &v43;
  v22 = sub_3411630(v16, 50, (int)a2 + 24, v37, v38, v37, v35);
  v24 = v23;
  v25 = v22;
  v40 = v22;
  *(_QWORD *)&v26 = sub_33EE280(*(_QWORD *)(a1 + 864), a2[1], v42, 0, 0, 0);
  *((_QWORD *)&v34 + 1) = v24;
  *(_QWORD *)&v34 = v25;
  *((_QWORD *)&v33 + 1) = 1;
  *(_QWORD *)&v33 = v40;
  v27 = sub_340F900(*(_QWORD *)(a1 + 864), 303, (int)a2 + 24, 1, 0, *(_QWORD *)(a1 + 864), v33, v26, v34);
  v29 = *(_QWORD *)(a1 + 864);
  v30 = v27;
  v31 = v28;
  if ( v27 )
  {
    nullsub_1875(v27, *(_QWORD *)(a1 + 864), 0);
    *(_QWORD *)(v29 + 384) = v30;
    *(_DWORD *)(v29 + 392) = v31;
    return sub_33E2B60(v29, 0);
  }
  else
  {
    *(_QWORD *)(v29 + 384) = 0;
    *(_DWORD *)(v29 + 392) = v28;
    return v28;
  }
}
