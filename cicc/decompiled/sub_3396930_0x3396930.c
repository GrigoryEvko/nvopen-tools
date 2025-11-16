// Function: sub_3396930
// Address: 0x3396930
//
void __fastcall sub_3396930(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int128 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // r15
  __int64 v9; // r14
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // r15
  int v13; // edx
  __int64 v14; // rax
  int v15; // edx
  unsigned __int16 v16; // ax
  __int128 v17; // rax
  int v18; // r9d
  __int64 v19; // r14
  int v20; // edx
  int v21; // r15d
  _QWORD *v22; // rax
  __int64 v23; // rsi
  __int64 (__fastcall *v24)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-88h]
  int v25; // [rsp+10h] [rbp-80h]
  int v26; // [rsp+18h] [rbp-78h]
  __int128 v27; // [rsp+20h] [rbp-70h]
  __int64 v28; // [rsp+48h] [rbp-48h] BYREF
  __int64 v29; // [rsp+50h] [rbp-40h] BYREF
  int v30; // [rsp+58h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  *(_QWORD *)&v4 = sub_338B750(a1, *v3);
  v29 = 0;
  v27 = v4;
  v5 = *(_QWORD *)a1;
  v30 = *(_DWORD *)(a1 + 848);
  if ( v5 )
  {
    if ( &v29 != (__int64 *)(v5 + 48) )
    {
      v6 = *(_QWORD *)(v5 + 48);
      v29 = v6;
      if ( v6 )
        sub_B96E90((__int64)&v29, v6, 1);
    }
  }
  v7 = *(_QWORD *)(a1 + 864);
  v8 = *(__int64 **)(a2 + 8);
  v9 = *(_QWORD *)(v7 + 16);
  v10 = sub_2E79000(*(__int64 **)(v7 + 40));
  v11 = sub_2D5BAE0(v9, v10, v8, 0);
  v12 = *(_QWORD *)(a1 + 864);
  v25 = v11;
  v26 = v13;
  v24 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v9 + 32LL);
  v14 = sub_2E79000(*(__int64 **)(v12 + 40));
  if ( v24 == sub_2D42F30 )
  {
    v15 = sub_AE2980(v14, 0)[1];
    v16 = 2;
    if ( v15 != 1 )
    {
      v16 = 3;
      if ( v15 != 2 )
      {
        v16 = 4;
        if ( v15 != 4 )
        {
          v16 = 5;
          if ( v15 != 8 )
          {
            v16 = 6;
            if ( v15 != 16 )
            {
              v16 = 7;
              if ( v15 != 32 )
              {
                v16 = 8;
                if ( v15 != 64 )
                  v16 = 9 * (v15 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v16 = v24(v9, v14, 0);
  }
  *(_QWORD *)&v17 = sub_3400BD0(v12, 0, (unsigned int)&v29, v16, 0, 1, 0);
  v19 = sub_3406EB0(v12, 230, (unsigned int)&v29, v25, v26, v18, v27, v17);
  v21 = v20;
  v28 = a2;
  v22 = sub_337DC20(a1 + 8, &v28);
  *v22 = v19;
  v23 = v29;
  *((_DWORD *)v22 + 2) = v21;
  if ( v23 )
    sub_B91220((__int64)&v29, v23);
}
