// Function: sub_3397E10
// Address: 0x3397e10
//
void __fastcall sub_3397E10(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 *v5; // rdx
  __int128 v6; // rax
  __int64 v7; // r12
  __int64 (__fastcall *v8)(__int64, __int64, __int64, __int64); // r13
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // rax
  int v12; // edx
  unsigned __int16 v13; // ax
  int v14; // edx
  unsigned int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rax
  int v25; // eax
  int v26; // edx
  int v27; // r9d
  int v28; // r15d
  __int64 v29; // rax
  int v30; // r8d
  bool v31; // zf
  __int64 v32; // rsi
  __int64 v33; // r12
  int v34; // edx
  int v35; // r13d
  _QWORD *v36; // rax
  __int128 v37; // [rsp-10h] [rbp-A0h]
  __int64 *v38; // [rsp+8h] [rbp-88h]
  int v39; // [rsp+8h] [rbp-88h]
  __int128 v40; // [rsp+10h] [rbp-80h]
  __int64 v41; // [rsp+20h] [rbp-70h]
  __int64 v42; // [rsp+48h] [rbp-48h] BYREF
  __int64 v43; // [rsp+50h] [rbp-40h] BYREF
  int v44; // [rsp+58h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  *(_QWORD *)&v6 = sub_338B750(a1, *v5);
  v7 = *(_QWORD *)(a1 + 864);
  v40 = v6;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v4 + 72LL);
  v9 = sub_2E79000(*(__int64 **)(v7 + 40));
  v10 = v9;
  if ( v8 == sub_2FE4D20 )
  {
    v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v4 + 32LL);
    if ( v11 == sub_2D42F30 )
    {
      v12 = sub_AE2980(v10, 0)[1];
      v13 = 2;
      if ( v12 != 1 )
      {
        v13 = 3;
        if ( v12 != 2 )
        {
          v13 = 4;
          if ( v12 != 4 )
          {
            v13 = 5;
            if ( v12 != 8 )
            {
              v13 = 6;
              if ( v12 != 16 )
              {
                v13 = 7;
                if ( v12 != 32 )
                {
                  v13 = 8;
                  if ( v12 != 64 )
                    v13 = 9 * (v12 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v13 = v11(v4, v10, 0);
    }
  }
  else
  {
    v13 = ((__int64 (__fastcall *)(__int64, __int64))v8)(v4, v9);
  }
  v14 = *(_DWORD *)(a1 + 848);
  v43 = 0;
  v15 = v13;
  v16 = *(_QWORD *)a1;
  v44 = v14;
  if ( v16 )
  {
    if ( &v43 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v43 = v17;
      if ( v17 )
        sub_B96E90((__int64)&v43, v17, 1);
    }
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v18 = *(_QWORD *)(a2 - 8);
  else
    v18 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v19 = sub_338B750(a1, *(_QWORD *)(v18 + 32));
  v21 = sub_33FB310(v7, v19, v20, &v43, v15, 0);
  v23 = v22;
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  v41 = *(_QWORD *)(a1 + 864);
  v38 = *(__int64 **)(a2 + 8);
  v24 = sub_2E79000(*(__int64 **)(v41 + 40));
  v25 = sub_2D5BAE0(v4, v24, v38, 0);
  v43 = 0;
  v27 = v41;
  v28 = v25;
  v29 = *(_QWORD *)a1;
  v30 = v26;
  v31 = *(_QWORD *)a1 == 0;
  v44 = *(_DWORD *)(a1 + 848);
  if ( !v31 && &v43 != (__int64 *)(v29 + 48) )
  {
    v32 = *(_QWORD *)(v29 + 48);
    v43 = v32;
    if ( v32 )
    {
      v39 = v26;
      sub_B96E90((__int64)&v43, v32, 1);
      v30 = v39;
      v27 = v41;
    }
  }
  *((_QWORD *)&v37 + 1) = v23;
  *(_QWORD *)&v37 = v21;
  v33 = sub_3406EB0(v27, 158, (unsigned int)&v43, v28, v30, v27, v40, v37);
  v35 = v34;
  v42 = a2;
  v36 = sub_337DC20(a1 + 8, &v42);
  *v36 = v33;
  *((_DWORD *)v36 + 2) = v35;
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
}
