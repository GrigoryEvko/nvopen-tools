// Function: sub_359B620
// Address: 0x359b620
//
__int64 __fastcall sub_359B620(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // r15
  int v6; // r15d
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r13
  int v12; // esi
  __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // esi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rbx
  int v18; // eax
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int32 v21; // eax
  unsigned __int8 *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r13
  unsigned __int8 *v25; // r14
  _QWORD *v26; // r13
  __int64 *v27; // rax
  unsigned __int8 *v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rbx
  unsigned int v31; // r13d
  __int64 v32; // r14
  __int64 v33; // r12
  __int64 v34; // rax
  unsigned int v35; // ebx
  __int64 v36; // r13
  __int64 v37; // r14
  __int64 v38; // rdx
  __int64 v39; // [rsp+8h] [rbp-C8h]
  __int64 v41; // [rsp+18h] [rbp-B8h]
  __int64 v42; // [rsp+30h] [rbp-A0h]
  _QWORD *v44; // [rsp+40h] [rbp-90h]
  __int64 v45; // [rsp+48h] [rbp-88h]
  __int64 v46; // [rsp+50h] [rbp-80h]
  __int64 v47; // [rsp+50h] [rbp-80h]
  unsigned __int32 v48; // [rsp+58h] [rbp-78h]
  unsigned __int32 v49; // [rsp+5Ch] [rbp-74h]
  __int64 v50; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int8 *v51; // [rsp+68h] [rbp-68h] BYREF
  unsigned __int8 *v52; // [rsp+70h] [rbp-60h] BYREF
  _QWORD *v53; // [rsp+78h] [rbp-58h]
  unsigned __int8 *v54; // [rsp+80h] [rbp-50h] BYREF
  __int64 v55; // [rsp+88h] [rbp-48h]
  __int64 v56; // [rsp+90h] [rbp-40h]

  v3 = a2;
  v44 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1[1] + 16LL) + 200LL))(*(_QWORD *)(a1[1] + 16LL));
  result = sub_2E311E0(a2);
  v5 = *(_QWORD *)(a2 + 56);
  v42 = a2 + 48;
  v41 = result;
  v50 = v5;
  if ( v5 != result )
  {
    v45 = v5;
    do
    {
      v6 = *(_DWORD *)(*(_QWORD *)(v45 + 32) + 8LL);
      v7 = a1[3];
      if ( v6 < 0 )
        v8 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 16LL * (v6 & 0x7FFFFFFF) + 8);
      else
        v8 = *(_QWORD *)(*(_QWORD *)(v7 + 304) + 8LL * (unsigned int)v6);
      if ( v8 )
      {
        while ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 )
        {
          v8 = *(_QWORD *)(v8 + 32);
          if ( !v8 )
            goto LABEL_15;
        }
        v9 = *(_QWORD *)(v8 + 16);
        v48 = *(_DWORD *)(*(_QWORD *)(v45 + 32) + 8LL);
        v10 = v3;
        v11 = v8;
LABEL_8:
        if ( *(_WORD *)(v9 + 68) != 68 && *(_WORD *)(v9 + 68) )
          goto LABEL_13;
        if ( v10 != *(_QWORD *)(v9 + 24) )
          goto LABEL_13;
        v12 = *(_DWORD *)(v45 + 40) & 0xFFFFFF;
        if ( v12 == 1 )
          goto LABEL_13;
        v13 = *(_QWORD *)(v45 + 32);
        v14 = 1;
        while ( v10 != *(_QWORD *)(v13 + 40LL * (unsigned int)(v14 + 1) + 24) )
        {
          v14 = (unsigned int)(v14 + 2);
          if ( v12 == (_DWORD)v14 )
            goto LABEL_13;
        }
        v15 = *(_DWORD *)(v13 + 40 * v14 + 8);
        if ( v15 )
        {
          v16 = sub_2EBEE10(a1[3], v15);
          v17 = v16;
          if ( !v16 )
            goto LABEL_29;
          if ( *(_QWORD *)(v16 + 24) != v10 )
            goto LABEL_29;
          v18 = *(unsigned __int16 *)(v16 + 68);
          if ( v18 == 68 )
            goto LABEL_29;
          if ( !v18 )
            goto LABEL_29;
          v46 = v17;
          v49 = 0;
          if ( v17 == v42 )
            goto LABEL_29;
          v39 = v11;
          do
          {
            if ( (unsigned int)sub_2E89C70(v46, v48, 0, 0) != -1 )
            {
              if ( !v49 )
              {
                v21 = sub_2EC06C0(
                        a1[3],
                        *(_QWORD *)(*(_QWORD *)(a1[3] + 56LL) + 16LL * (v48 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                        byte_3F871B3,
                        0,
                        v19,
                        v20);
                v22 = *(unsigned __int8 **)(v17 + 56);
                v49 = v21;
                v23 = a1[4];
                v51 = v22;
                v24 = *(_QWORD *)(v23 + 8) - 800LL;
                if ( v22 )
                {
                  sub_B96E90((__int64)&v51, (__int64)v22, 1);
                  v54 = v51;
                  if ( v51 )
                  {
                    sub_B976B0((__int64)&v51, v51, (__int64)&v54);
                    v51 = 0;
                  }
                }
                else
                {
                  v54 = 0;
                }
                v55 = 0;
                v56 = 0;
                if ( (*(_BYTE *)(v17 + 44) & 4) != 0 )
                {
                  v25 = *(unsigned __int8 **)(v10 + 32);
                  v52 = v54;
                  if ( v54 )
                    sub_B96E90((__int64)&v52, (__int64)v54, 1);
                  v26 = sub_2E7B380(v25, v24, &v52, 0);
                  if ( v52 )
                    sub_B91220((__int64)&v52, (__int64)v52);
                  sub_2E326B0(v10, (__int64 *)v17, (__int64)v26);
                  v52 = v25;
                  v53 = v26;
                  if ( v55 )
                    sub_2E882B0((__int64)v26, (__int64)v25, v55);
                  if ( v56 )
                    sub_2E88680((__int64)v53, (__int64)v52, v56);
                  v27 = sub_3598AB0((__int64 *)&v52, v49, 2u, 0);
                  v28 = (unsigned __int8 *)*v27;
                  v29 = v27[1];
                }
                else
                {
                  v28 = (unsigned __int8 *)sub_2F26260(v10, (__int64 *)v17, (__int64 *)&v54, v24, v49);
                  v29 = v38;
                }
                v52 = v28;
                v53 = (_QWORD *)v29;
                sub_3598AB0((__int64 *)&v52, v48, 0, 0);
                if ( v54 )
                  sub_B91220((__int64)&v54, (__int64)v54);
                if ( v51 )
                  sub_B91220((__int64)&v51, (__int64)v51);
              }
              sub_2E8A790(v46, v48, v49, 0, v44);
            }
            v46 = *(_QWORD *)(v46 + 8);
          }
          while ( v42 != v46 );
          v11 = v39;
          if ( !v49 )
          {
LABEL_29:
            v9 = *(_QWORD *)(v11 + 16);
            goto LABEL_13;
          }
          v3 = v10;
          v30 = *a3;
          v47 = *a3 + 8LL * *((unsigned int *)a3 + 2);
          if ( v47 == *a3 )
            goto LABEL_15;
          v31 = v48;
          do
          {
            v32 = *(_QWORD *)(*(_QWORD *)v30 + 56LL);
            v33 = *(_QWORD *)v30 + 48LL;
            if ( v33 != v32 )
            {
              v34 = v30;
              v35 = v31;
              v36 = v32;
              v37 = v34;
              do
              {
                while ( 1 )
                {
                  if ( (unsigned int)sub_2E89C70(v36, v35, 0, 0) != -1 )
                    sub_2E8A790(v36, v35, v49, 0, v44);
                  if ( !v36 )
                    BUG();
                  if ( (*(_BYTE *)v36 & 4) == 0 )
                    break;
                  v36 = *(_QWORD *)(v36 + 8);
                  if ( v33 == v36 )
                    goto LABEL_67;
                }
                while ( (*(_BYTE *)(v36 + 44) & 8) != 0 )
                  v36 = *(_QWORD *)(v36 + 8);
                v36 = *(_QWORD *)(v36 + 8);
              }
              while ( v33 != v36 );
LABEL_67:
              v31 = v35;
              v30 = v37;
            }
            v30 += 8;
          }
          while ( v47 != v30 );
        }
        else
        {
LABEL_13:
          while ( 1 )
          {
            v11 = *(_QWORD *)(v11 + 32);
            if ( !v11 )
              break;
            if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 && *(_QWORD *)(v11 + 16) != v9 )
            {
              v9 = *(_QWORD *)(v11 + 16);
              goto LABEL_8;
            }
          }
        }
        v3 = v10;
      }
LABEL_15:
      sub_2FD79B0(&v50);
      result = v50;
      v45 = v50;
    }
    while ( v50 != v41 );
  }
  return result;
}
