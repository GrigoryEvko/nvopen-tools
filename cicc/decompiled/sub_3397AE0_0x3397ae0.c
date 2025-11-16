// Function: sub_3397AE0
// Address: 0x3397ae0
//
void __fastcall sub_3397AE0(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 *v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int128 v8; // rax
  __int64 v9; // r12
  __int64 (__fastcall *v10)(__int64, __int64, __int64, __int64); // r13
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int); // rax
  int v14; // edx
  unsigned __int16 v15; // ax
  int v16; // edx
  unsigned int v17; // r13d
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // rax
  int v27; // eax
  int v28; // edx
  int v29; // r9d
  int v30; // r15d
  __int64 v31; // rax
  int v32; // r8d
  bool v33; // zf
  __int64 v34; // rsi
  __int64 v35; // r12
  int v36; // edx
  int v37; // r13d
  _QWORD *v38; // rax
  __int128 v39; // [rsp-10h] [rbp-B0h]
  __int64 *v40; // [rsp+8h] [rbp-98h]
  int v41; // [rsp+8h] [rbp-98h]
  __int128 v42; // [rsp+10h] [rbp-90h]
  __int128 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+30h] [rbp-70h]
  __int64 v45; // [rsp+58h] [rbp-48h] BYREF
  __int64 v46; // [rsp+60h] [rbp-40h] BYREF
  int v47; // [rsp+68h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  *(_QWORD *)&v43 = sub_338B750(a1, *v5);
  *((_QWORD *)&v43 + 1) = v6;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a2 - 8);
  else
    v7 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  *(_QWORD *)&v8 = sub_338B750(a1, *(_QWORD *)(v7 + 32));
  v9 = *(_QWORD *)(a1 + 864);
  v42 = v8;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v4 + 72LL);
  v11 = sub_2E79000(*(__int64 **)(v9 + 40));
  v12 = v11;
  if ( v10 == sub_2FE4D20 )
  {
    v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v4 + 32LL);
    if ( v13 == sub_2D42F30 )
    {
      v14 = sub_AE2980(v12, 0)[1];
      v15 = 2;
      if ( v14 != 1 )
      {
        v15 = 3;
        if ( v14 != 2 )
        {
          v15 = 4;
          if ( v14 != 4 )
          {
            v15 = 5;
            if ( v14 != 8 )
            {
              v15 = 6;
              if ( v14 != 16 )
              {
                v15 = 7;
                if ( v14 != 32 )
                {
                  v15 = 8;
                  if ( v14 != 64 )
                    v15 = 9 * (v14 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v15 = v13(v4, v12, 0);
    }
  }
  else
  {
    v15 = ((__int64 (__fastcall *)(__int64, __int64))v10)(v4, v11);
  }
  v16 = *(_DWORD *)(a1 + 848);
  v46 = 0;
  v17 = v15;
  v18 = *(_QWORD *)a1;
  v47 = v16;
  if ( v18 )
  {
    if ( &v46 != (__int64 *)(v18 + 48) )
    {
      v19 = *(_QWORD *)(v18 + 48);
      v46 = v19;
      if ( v19 )
        sub_B96E90((__int64)&v46, v19, 1);
    }
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v20 = *(_QWORD *)(a2 - 8);
  else
    v20 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v21 = sub_338B750(a1, *(_QWORD *)(v20 + 64));
  v23 = sub_33FB310(v9, v21, v22, &v46, v17, 0);
  v25 = v24;
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
  v44 = *(_QWORD *)(a1 + 864);
  v40 = *(__int64 **)(a2 + 8);
  v26 = sub_2E79000(*(__int64 **)(v44 + 40));
  v27 = sub_2D5BAE0(v4, v26, v40, 0);
  v46 = 0;
  v29 = v44;
  v30 = v27;
  v31 = *(_QWORD *)a1;
  v32 = v28;
  v33 = *(_QWORD *)a1 == 0;
  v47 = *(_DWORD *)(a1 + 848);
  if ( !v33 && &v46 != (__int64 *)(v31 + 48) )
  {
    v34 = *(_QWORD *)(v31 + 48);
    v46 = v34;
    if ( v34 )
    {
      v41 = v28;
      sub_B96E90((__int64)&v46, v34, 1);
      v32 = v41;
      v29 = v44;
    }
  }
  *((_QWORD *)&v39 + 1) = v25;
  *(_QWORD *)&v39 = v23;
  v35 = sub_340F900(v29, 157, (unsigned int)&v46, v30, v32, v29, v43, v42, v39);
  v37 = v36;
  v45 = a2;
  v38 = sub_337DC20(a1 + 8, &v45);
  *v38 = v35;
  *((_DWORD *)v38 + 2) = v37;
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
}
