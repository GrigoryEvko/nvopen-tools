// Function: sub_337E7C0
// Address: 0x337e7c0
//
void __fastcall sub_337E7C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 *v13; // rdi
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int); // rax
  int v18; // edx
  unsigned __int16 v19; // ax
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 *v22; // rdi
  int v23; // edx
  __int64 v24; // rdi
  __int64 (__fastcall *v25)(__int64, __int64, unsigned int); // rax
  int v26; // edx
  unsigned __int16 v27; // ax
  __int64 v28; // rax
  __int64 v29; // rdi
  int v30; // edx
  int v31; // r9d
  __int64 v32; // r12
  int v33; // edx
  int v34; // r15d
  _QWORD *v35; // rax
  __int64 v36; // r13
  __int128 v37; // [rsp-20h] [rbp-110h]
  __int64 (__fastcall *v38)(__int64, __int64, __int64, __int64); // [rsp+8h] [rbp-E8h]
  __int64 (__fastcall *v39)(__int64, __int64, __int64, __int64); // [rsp+8h] [rbp-E8h]
  __int64 v40; // [rsp+78h] [rbp-78h] BYREF
  __int64 v41; // [rsp+80h] [rbp-70h] BYREF
  int v42; // [rsp+88h] [rbp-68h]
  __int128 v43; // [rsp+90h] [rbp-60h] BYREF
  __int128 v44; // [rsp+A0h] [rbp-50h]
  __int128 v45; // [rsp+B0h] [rbp-40h]

  v6 = a2;
  v8 = *(unsigned int *)(a1 + 848);
  v9 = *(_QWORD *)a1;
  v41 = 0;
  v42 = v8;
  if ( v9 )
  {
    v8 = v9 + 48;
    if ( &v41 != (__int64 *)(v9 + 48) )
    {
      a2 = *(_QWORD *)(v9 + 48);
      v41 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v41, a2, 1);
    }
  }
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v11 = sub_33738B0(a1, a2, v8, a4, a5, a6);
  v12 = *(_QWORD *)(a1 + 864);
  v13 = *(__int64 **)(v12 + 40);
  *(_QWORD *)&v43 = v11;
  DWORD2(v43) = v14;
  v38 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v10 + 48LL);
  v15 = sub_2E79000(v13);
  v16 = v15;
  if ( v38 == sub_2FE4DB0 )
  {
    v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v10 + 32LL);
    if ( v17 == sub_2D42F30 )
    {
      v18 = sub_AE2980(v16, 0)[1];
      v19 = 2;
      if ( v18 != 1 )
      {
        v19 = 3;
        if ( v18 != 2 )
        {
          v19 = 4;
          if ( v18 != 4 )
          {
            v19 = 5;
            if ( v18 != 8 )
            {
              v19 = 6;
              if ( v18 != 16 )
              {
                v19 = 7;
                if ( v18 != 32 )
                {
                  v19 = 8;
                  if ( v18 != 64 )
                    v19 = 9 * (v18 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v19 = v17(v10, v16, 0);
    }
  }
  else
  {
    v19 = ((__int64 (__fastcall *)(__int64, __int64))v38)(v10, v15);
  }
  v20 = sub_3400BD0(v12, *(_WORD *)(v6 + 2) & 7, (unsigned int)&v41, v19, 0, 1, 0);
  v21 = *(_QWORD *)(a1 + 864);
  v22 = *(__int64 **)(v21 + 40);
  *(_QWORD *)&v44 = v20;
  DWORD2(v44) = v23;
  v39 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v10 + 48LL);
  v24 = sub_2E79000(v22);
  if ( v39 == sub_2FE4DB0 )
  {
    v25 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v10 + 32LL);
    if ( v25 == sub_2D42F30 )
    {
      v26 = sub_AE2980(v24, 0)[1];
      v27 = 2;
      if ( v26 != 1 )
      {
        v27 = 3;
        if ( v26 != 2 )
        {
          v27 = 4;
          if ( v26 != 4 )
          {
            v27 = 5;
            if ( v26 != 8 )
            {
              v27 = 6;
              if ( v26 != 16 )
              {
                v27 = 7;
                if ( v26 != 32 )
                {
                  v27 = 8;
                  if ( v26 != 64 )
                    v27 = 9 * (v26 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v27 = v25(v10, v24, 0);
    }
  }
  else
  {
    v27 = ((__int64 (__fastcall *)(__int64, __int64))v39)(v10, v24);
  }
  v28 = sub_3400BD0(v21, *(unsigned __int8 *)(v6 + 72), (unsigned int)&v41, v27, 0, 1, 0);
  v29 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v45 = v28;
  *((_QWORD *)&v37 + 1) = 3;
  DWORD2(v45) = v30;
  *(_QWORD *)&v37 = &v43;
  v32 = sub_33FC220(v29, 337, (unsigned int)&v41, 1, 0, v31, v37);
  v34 = v33;
  v40 = v6;
  v35 = sub_337DC20(a1 + 8, &v40);
  *v35 = v32;
  *((_DWORD *)v35 + 2) = v34;
  v36 = *(_QWORD *)(a1 + 864);
  if ( v32 )
  {
    nullsub_1875(v32, *(_QWORD *)(a1 + 864), 0);
    *(_QWORD *)(v36 + 384) = v32;
    *(_DWORD *)(v36 + 392) = v34;
    sub_33E2B60(v36, 0);
  }
  else
  {
    *(_QWORD *)(v36 + 384) = 0;
    *(_DWORD *)(v36 + 392) = v34;
  }
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
}
