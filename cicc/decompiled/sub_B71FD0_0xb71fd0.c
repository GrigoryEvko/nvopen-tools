// Function: sub_B71FD0
// Address: 0xb71fd0
//
void __fastcall sub_B71FD0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // r14
  __int64 v3; // rbx
  _QWORD **v4; // r13
  _QWORD *v5; // r12
  __int64 v6; // r8
  __int64 v7; // rsi
  __int64 v8; // r11
  __int64 v9; // r9
  __int64 v10; // r10
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+10h] [rbp-70h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  __int64 v22; // [rsp+20h] [rbp-60h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+28h] [rbp-58h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+30h] [rbp-50h]
  __int64 v31; // [rsp+30h] [rbp-50h]
  __int64 v32; // [rsp+30h] [rbp-50h]
  __int64 v33; // [rsp+30h] [rbp-50h]
  __int64 v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+38h] [rbp-48h]
  __int64 v36; // [rsp+38h] [rbp-48h]
  __int64 v37; // [rsp+38h] [rbp-48h]
  __int64 v38; // [rsp+38h] [rbp-48h]
  __int64 v39; // [rsp+38h] [rbp-48h]
  __int64 v40; // [rsp+40h] [rbp-40h]
  __int64 v41; // [rsp+48h] [rbp-38h]
  __int64 v42; // [rsp+48h] [rbp-38h]
  __int64 v43; // [rsp+48h] [rbp-38h]
  __int64 v44; // [rsp+48h] [rbp-38h]
  __int64 v45; // [rsp+48h] [rbp-38h]

  v1 = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)v1 )
  {
    v2 = 8 * v1;
    v3 = 0;
    do
    {
      v4 = (_QWORD **)(v3 + *(_QWORD *)a1);
      v5 = *v4;
      if ( *v4 != (_QWORD *)-8LL && v5 )
      {
        v6 = v5[1];
        v7 = *v5 + 17LL;
        if ( v6 )
        {
          v8 = *(_QWORD *)(v6 + 32);
          if ( v8 )
          {
            v9 = *(_QWORD *)(v8 + 32);
            if ( v9 )
            {
              v10 = *(_QWORD *)(v9 + 32);
              if ( v10 )
              {
                v11 = *(_QWORD *)(v10 + 32);
                v41 = v11;
                if ( v11 )
                {
                  v12 = *(_QWORD *)(v11 + 32);
                  v40 = v12;
                  if ( v12 )
                  {
                    v13 = *(_QWORD *)(v12 + 32);
                    if ( v13 )
                    {
                      v14 = *(_QWORD *)(v13 + 32);
                      if ( v14 )
                      {
                        v15 = v13;
                        v16 = *(_QWORD *)(v9 + 32);
                        v18 = *(_QWORD *)(v8 + 32);
                        v21 = *(_QWORD *)(v6 + 32);
                        v25 = v5[1];
                        v34 = *(_QWORD *)(v13 + 32);
                        sub_AC5B80((__int64 *)(v14 + 32));
                        sub_BD7260(v34);
                        sub_BD2DD0(v34);
                        v13 = v15;
                        v10 = v16;
                        v9 = v18;
                        v8 = v21;
                        v6 = v25;
                      }
                      v17 = v10;
                      v19 = v9;
                      v22 = v8;
                      v26 = v6;
                      v35 = v13;
                      sub_BD7260(v13);
                      sub_BD2DD0(v35);
                      v10 = v17;
                      v9 = v19;
                      v8 = v22;
                      v6 = v26;
                    }
                    v20 = v10;
                    v23 = v9;
                    v27 = v8;
                    v30 = v6;
                    sub_BD7260(v40);
                    sub_BD2DD0(v40);
                    v10 = v20;
                    v9 = v23;
                    v8 = v27;
                    v6 = v30;
                  }
                  v24 = v10;
                  v28 = v9;
                  v31 = v8;
                  v36 = v6;
                  sub_BD7260(v41);
                  sub_BD2DD0(v41);
                  v10 = v24;
                  v9 = v28;
                  v8 = v31;
                  v6 = v36;
                }
                v29 = v9;
                v32 = v8;
                v37 = v6;
                v42 = v10;
                sub_BD7260(v10);
                sub_BD2DD0(v42);
                v9 = v29;
                v8 = v32;
                v6 = v37;
              }
              v33 = v8;
              v38 = v6;
              v43 = v9;
              sub_BD7260(v9);
              sub_BD2DD0(v43);
              v8 = v33;
              v6 = v38;
            }
            v39 = v6;
            v44 = v8;
            sub_BD7260(v8);
            sub_BD2DD0(v44);
            v6 = v39;
          }
          v45 = v6;
          sub_BD7260(v6);
          sub_BD2DD0(v45);
        }
        sub_C7D6A0(v5, v7, 8);
      }
      v3 += 8;
      *v4 = 0;
    }
    while ( v2 != v3 );
  }
  *(_QWORD *)(a1 + 12) = 0;
}
