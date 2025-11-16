// Function: sub_16E9280
// Address: 0x16e9280
//
__int64 __fastcall sub_16E9280(__int64 a1, __int64 a2, int a3, int a4, int a5)
{
  __int64 result; // rax
  __int64 v6; // r9
  int v8; // r14d
  int v10; // eax
  int v11; // edx
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // r12
  signed __int64 v15; // rdx
  __int64 v16; // rcx
  signed __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 *v19; // rcx
  signed __int64 v20; // rax
  signed __int64 v21; // rdx
  __int64 v22; // rcx
  signed __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 *v25; // rdx
  signed __int64 v26; // rdx
  __int64 v27; // rcx
  signed __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 *v30; // rcx
  int v31; // r8d
  __int64 v32; // r9
  signed __int64 v33; // r15
  signed __int64 v34; // rdx
  signed __int64 v35; // rax
  __int64 v36; // rcx
  signed __int64 v37; // rsi
  __int64 v38; // rdx
  signed __int64 v39; // rax
  signed __int64 v40; // rdx
  __int64 v41; // rcx
  signed __int64 v42; // rsi
  __int64 v43; // rdx
  signed __int64 v44; // rax
  signed __int64 v45; // rdx
  __int64 v46; // rcx
  signed __int64 v47; // rsi
  __int64 v48; // rdx
  int v49; // r8d
  int v50; // r9d
  __int64 v51; // r12
  signed __int64 v52; // rdx
  __int64 v53; // rcx
  signed __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // [rsp-40h] [rbp-40h]
  __int64 v57; // [rsp-40h] [rbp-40h]
  __int64 v58; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 40);
  if ( !(_DWORD)result )
  {
    v8 = a4;
LABEL_3:
    v10 = 8 * a3;
    if ( a3 > 1 )
      v10 = 8 * (a3 == 256) + 16;
    while ( 2 )
    {
      v11 = v8;
      if ( v8 > 1 )
        v11 = (v8 == 256) + 2;
      result = (unsigned int)(v11 + v10);
      switch ( (int)result )
      {
        case 0:
          *(_QWORD *)(a1 + 40) = a2;
          return result;
        case 1:
        case 2:
        case 3:
          sub_16E9180((_QWORD *)a1, 2013265920, v6 - a2 + 1, a2, a5, v6 - a2);
          result = sub_16E9280(a1, a2 + 1, 1, (unsigned int)v8);
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v14 = *(_QWORD *)(a1 + 40);
            v15 = *(_QWORD *)(a1 + 32);
            result = v14;
            if ( v14 >= v15 )
            {
              v16 = (v15 + 1) / 2;
              v17 = v16 + ((v15 + 1 + ((unsigned __int64)(v15 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v15 < v17 )
              {
                sub_16E90A0(a1, v17, v15, v16, v12, v13);
                result = *(_QWORD *)(a1 + 40);
              }
            }
            v18 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = result + 1;
            *(_QWORD *)(v18 + 8 * result) = (v14 - a2) | 0x80000000LL;
            if ( !*(_DWORD *)(a1 + 16) )
            {
              v19 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8 * a2);
              result = (*(_QWORD *)(a1 + 40) - a2) | *v19 & 0xF8000000LL;
              *v19 = result;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v20 = *(_QWORD *)(a1 + 40);
                v21 = *(_QWORD *)(a1 + 32);
                if ( v20 >= v21 )
                {
                  v22 = (v21 + 1) / 2;
                  v23 = v22 + ((v21 + 1 + ((unsigned __int64)(v21 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v21 < v23 )
                  {
                    sub_16E90A0(a1, v23, v21, v22, v12, v13);
                    v20 = *(_QWORD *)(a1 + 40);
                  }
                }
                v24 = *(_QWORD *)(a1 + 24);
                *(_QWORD *)(a1 + 40) = v20 + 1;
                *(_QWORD *)(v24 + 8 * v20) = 2281701376LL;
                result = *(_QWORD *)(a1 + 40) - 1LL;
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  v25 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8 * result);
                  result = *v25 & 0xF8000000LL | 1;
                  *v25 = result;
                  if ( !*(_DWORD *)(a1 + 16) )
                  {
                    result = *(_QWORD *)(a1 + 40);
                    v26 = *(_QWORD *)(a1 + 32);
                    if ( result >= v26 )
                    {
                      v27 = (v26 + 1) / 2;
                      v28 = v27 + ((v26 + 1 + ((unsigned __int64)(v26 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                      if ( v26 < v28 )
                      {
                        sub_16E90A0(a1, v28, v26, v27, v12, v13);
                        result = *(_QWORD *)(a1 + 40);
                      }
                    }
                    v29 = *(_QWORD *)(a1 + 24);
                    *(_QWORD *)(a1 + 40) = result + 1;
                    *(_QWORD *)(v29 + 8 * result) = 2415919106LL;
                  }
                }
              }
            }
          }
          return result;
        case 9:
          return result;
        case 10:
          v56 = v6;
          sub_16E9180((_QWORD *)a1, 2013265920, v6 - a2 + 1, a2, a5, v6);
          v32 = v56;
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v33 = *(_QWORD *)(a1 + 40);
            v34 = *(_QWORD *)(a1 + 32);
            v35 = v33;
            if ( v33 >= v34 )
            {
              v36 = (v34 + 1) / 2;
              v37 = v36 + ((v34 + 1 + ((unsigned __int64)(v34 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v34 < v37 )
              {
                sub_16E90A0(a1, v37, v34, v36, v31, v56);
                v35 = *(_QWORD *)(a1 + 40);
                v32 = v56;
              }
            }
            v38 = *(_QWORD *)(a1 + 24);
            LODWORD(v30) = v35 + 1;
            *(_QWORD *)(a1 + 40) = v35 + 1;
            *(_QWORD *)(v38 + 8 * v35) = (v33 - a2) | 0x80000000LL;
            v31 = *(_DWORD *)(a1 + 16);
            if ( !v31 )
            {
              v30 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8 * a2);
              *v30 = (*(_QWORD *)(a1 + 40) - a2) | *v30 & 0xF8000000LL;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v39 = *(_QWORD *)(a1 + 40);
                v40 = *(_QWORD *)(a1 + 32);
                if ( v39 >= v40 )
                {
                  v41 = (v40 + 1) / 2;
                  v42 = v41 + ((v40 + 1 + ((unsigned __int64)(v40 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v40 < v42 )
                  {
                    v57 = v32;
                    sub_16E90A0(a1, v42, v40, v41, 0, v32);
                    v39 = *(_QWORD *)(a1 + 40);
                    v32 = v57;
                  }
                }
                v43 = *(_QWORD *)(a1 + 24);
                LODWORD(v30) = v39 + 1;
                *(_QWORD *)(a1 + 40) = v39 + 1;
                *(_QWORD *)(v43 + 8 * v39) = 2281701376LL;
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * (*(_QWORD *)(a1 + 40) - 1LL)) = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * (*(_QWORD *)(a1 + 40) - 1LL))
                                                                                       & 0xF8000000LL
                                                                                       | 1;
                  LODWORD(v30) = *(_DWORD *)(a1 + 16);
                  if ( !(_DWORD)v30 )
                  {
                    v44 = *(_QWORD *)(a1 + 40);
                    v45 = *(_QWORD *)(a1 + 32);
                    if ( v44 >= v45 )
                    {
                      v46 = (v45 + 1) / 2;
                      v47 = v46 + ((v45 + 1 + ((unsigned __int64)(v45 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                      if ( v45 < v47 )
                      {
                        v58 = v32;
                        sub_16E90A0(a1, v47, v45, v46, v31, v32);
                        v44 = *(_QWORD *)(a1 + 40);
                        v32 = v58;
                      }
                    }
                    v48 = *(_QWORD *)(a1 + 24);
                    LODWORD(v30) = v44 + 1;
                    *(_QWORD *)(a1 + 40) = v44 + 1;
                    *(_QWORD *)(v48 + 8 * v44) = 2415919106LL;
                  }
                }
              }
            }
          }
          --v8;
          result = sub_16E9110((_QWORD *)a1, a2 + 1, v32 + 1, (int)v30, v31, v32);
          v6 = *(_QWORD *)(a1 + 40);
          a2 = result;
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v10 = 8;
            a3 = 1;
            continue;
          }
          return result;
        case 11:
          sub_16E9180((_QWORD *)a1, 1207959552, v6 - a2 + 1, a2, a5, v6 - a2);
          result = *(unsigned int *)(a1 + 16);
          if ( !(_DWORD)result )
          {
            v51 = *(_QWORD *)(a1 + 40);
            v52 = *(_QWORD *)(a1 + 32);
            result = v51;
            if ( v51 >= v52 )
            {
              v53 = (v52 + 1) / 2;
              v54 = v53 + ((v52 + 1 + ((unsigned __int64)(v52 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v52 < v54 )
              {
                sub_16E90A0(a1, v54, v52, v53, v49, v50);
                result = *(_QWORD *)(a1 + 40);
              }
            }
            v55 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = result + 1;
            *(_QWORD *)(v55 + 8 * result) = (v51 - a2) | 0x50000000;
          }
          return result;
        case 18:
          --v8;
          --a3;
          a2 = sub_16E9110((_QWORD *)a1, a2, v6, a4, a5, v6);
          goto LABEL_48;
        case 19:
          --a3;
          a2 = sub_16E9110((_QWORD *)a1, a2, v6, a4, a5, v6);
LABEL_48:
          result = *(unsigned int *)(a1 + 16);
          v6 = *(_QWORD *)(a1 + 40);
          if ( (_DWORD)result )
            return result;
          goto LABEL_3;
        default:
          result = (__int64)&unk_4FA17D0;
          *(_DWORD *)(a1 + 16) = 15;
          *(_QWORD *)a1 = &unk_4FA17D0;
          *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
          return result;
      }
    }
  }
  return result;
}
