// Function: sub_1464220
// Address: 0x1464220
//
void __fastcall sub_1464220(__int64 a1, __int64 a2)
{
  int v2; // r15d
  int v5; // r15d
  char v6; // r8
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r15
  __int64 *v11; // rax
  char v12; // r9
  __int64 *v13; // rax
  __int64 v14; // rsi
  char *v15; // rax
  char *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rdx
  char v23; // r8
  __int64 *v24; // rax
  __int64 v25; // rsi
  char *v26; // rax
  char *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rsi
  char v30; // al
  _QWORD *v31; // r12
  __int64 v32; // r9
  int v33; // r10d
  char v34; // [rsp+7h] [rbp-99h]
  char v35; // [rsp+7h] [rbp-99h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+8h] [rbp-98h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 *v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+8h] [rbp-98h]
  __int64 *v41; // [rsp+8h] [rbp-98h]
  __int64 *v42; // [rsp+10h] [rbp-90h] BYREF
  _BYTE v43[16]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v44; // [rsp+28h] [rbp-78h]
  _QWORD *v45; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v46[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v47; // [rsp+58h] [rbp-48h]
  __int64 v48; // [rsp+60h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 168);
  if ( v2 )
  {
    v5 = v2 - 1;
    v36 = *(_QWORD *)(a1 + 152);
    sub_1457D90(&v42, -8, 0);
    sub_1457D90(&v45, -16, 0);
    v6 = 1;
    v7 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = v36 + 48LL * v7;
    v9 = *(_QWORD *)(v8 + 24);
    if ( v9 != a2 )
    {
      v32 = v36 + 48LL * (v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)));
      v33 = 1;
      v8 = 0;
      while ( v9 != v44 )
      {
        if ( v9 != v47 || v8 )
          v32 = v8;
        v7 = v5 & (v33 + v7);
        v8 = v36 + 48LL * v7;
        v9 = *(_QWORD *)(v8 + 24);
        if ( v9 == a2 )
        {
          v6 = 1;
          goto LABEL_4;
        }
        ++v33;
        v8 = v32;
        v32 = v36 + 48LL * v7;
      }
      v6 = 0;
      if ( !v8 )
        v8 = v32;
    }
LABEL_4:
    v45 = &unk_49EE2B0;
    if ( v47 != 0 && v47 != -8 && v47 != -16 )
    {
      v34 = v6;
      v37 = v8;
      sub_1649B30(v46);
      v6 = v34;
      v8 = v37;
    }
    v42 = (__int64 *)&unk_49EE2B0;
    if ( v44 != 0 && v44 != -8 && v44 != -16 )
    {
      v35 = v6;
      v38 = v8;
      sub_1649B30(v43);
      v6 = v35;
      v8 = v38;
    }
    if ( v6 && v8 != *(_QWORD *)(a1 + 152) + 48LL * *(unsigned int *)(a1 + 168) )
    {
      v10 = *(_QWORD *)(v8 + 40);
      v11 = sub_1456EA0(a1, v10);
      if ( v11 )
      {
        v45 = (_QWORD *)a2;
        v46[0] = 0;
        v39 = v11;
        v12 = sub_14640F0((__int64)v11, (__int64 *)&v45, &v42);
        v13 = v42;
        if ( v12 )
        {
          *v42 = -16;
          v13[1] = -16;
          --*((_DWORD *)v39 + 4);
          v14 = v39[5];
          ++*((_DWORD *)v39 + 5);
          v15 = (char *)(sub_14530E0((_QWORD *)v39[4], v14, (__int64 *)&v45) + 2);
          v16 = (char *)v39[5];
          if ( v16 != v15 )
          {
            v17 = (v16 - v15) >> 4;
            if ( v16 - v15 <= 0 )
            {
              v15 = (char *)v39[5];
            }
            else
            {
              do
              {
                v18 = *(_QWORD *)v15;
                v15 += 16;
                *((_QWORD *)v15 - 4) = v18;
                *((_QWORD *)v15 - 3) = *((_QWORD *)v15 - 1);
                --v17;
              }
              while ( v17 );
              v15 = (char *)v39[5];
            }
          }
          v39[5] = (__int64)(v15 - 16);
        }
      }
      v19 = sub_1452BC0(v10);
      v40 = v20;
      if ( v20 )
      {
        v21 = sub_1456EA0(a1, v19);
        if ( v21 )
        {
          v22 = v40;
          v45 = (_QWORD *)a2;
          v41 = v21;
          v46[0] = v22;
          v23 = sub_14640F0((__int64)v21, (__int64 *)&v45, &v42);
          v24 = v42;
          if ( v23 )
          {
            *v42 = -16;
            v24[1] = -16;
            --*((_DWORD *)v41 + 4);
            v25 = v41[5];
            ++*((_DWORD *)v41 + 5);
            v26 = (char *)(sub_14530E0((_QWORD *)v41[4], v25, (__int64 *)&v45) + 2);
            v27 = (char *)v41[5];
            if ( v27 != v26 )
            {
              v28 = (v27 - v26) >> 4;
              if ( v27 - v26 <= 0 )
              {
                v26 = (char *)v41[5];
              }
              else
              {
                do
                {
                  v29 = *(_QWORD *)v26;
                  v26 += 16;
                  *((_QWORD *)v26 - 4) = v29;
                  *((_QWORD *)v26 - 3) = *((_QWORD *)v26 - 1);
                  --v28;
                }
                while ( v28 );
                v26 = (char *)v41[5];
              }
            }
            v41[5] = (__int64)(v26 - 16);
          }
        }
      }
      sub_1457D90(&v42, a2, 0);
      v30 = sub_145F6E0(a1 + 144, (__int64)&v42, &v45);
      v31 = v45;
      if ( v30 )
      {
        sub_1457D90(&v45, -16, 0);
        sub_1453650((__int64)(v31 + 1), v46);
        v31[4] = v48;
        v45 = &unk_49EE2B0;
        sub_1455FA0((__int64)v46);
        --*(_DWORD *)(a1 + 160);
        ++*(_DWORD *)(a1 + 164);
      }
      v42 = (__int64 *)&unk_49EE2B0;
      if ( v44 != 0 && v44 != -8 && v44 != -16 )
        sub_1649B30(v43);
    }
  }
}
