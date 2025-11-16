// Function: sub_CB7920
// Address: 0xcb7920
//
__int64 __fastcall sub_CB7920(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 result; // rax
  __int64 v5; // r9
  int v9; // eax
  int v10; // edx
  __int64 v11; // r12
  signed __int64 v12; // rdx
  signed __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 *v15; // rcx
  signed __int64 v16; // rax
  signed __int64 v17; // rdx
  signed __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 *v20; // rdx
  signed __int64 v21; // rdx
  signed __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // r9
  signed __int64 v25; // r15
  signed __int64 v26; // rdx
  signed __int64 v27; // rax
  signed __int64 v28; // rsi
  __int64 v29; // rdx
  signed __int64 v30; // rax
  signed __int64 v31; // rdx
  signed __int64 v32; // rsi
  __int64 v33; // rdx
  signed __int64 v34; // rax
  signed __int64 v35; // rdx
  signed __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // r12
  signed __int64 v39; // rdx
  signed __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]
  __int64 v44; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 16);
  v5 = *(_QWORD *)(a1 + 40);
  if ( !(_DWORD)result )
  {
LABEL_2:
    v9 = 8 * a3;
    if ( a3 > 1 )
      v9 = 8 * (a3 == 256) + 16;
    while ( 2 )
    {
      v10 = a4;
      if ( a4 > 1 )
        v10 = (a4 == 256) + 2;
      result = (unsigned int)(v10 + v9);
      switch ( (int)result )
      {
        case 0:
          *(_QWORD *)(a1 + 40) = a2;
          return result;
        case 1:
        case 2:
        case 3:
          sub_CB7820((_QWORD *)a1, 2013265920, v5 - a2 + 1, a2);
          result = sub_CB7920(a1, a2 + 1, 1, (unsigned int)a4);
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v11 = *(_QWORD *)(a1 + 40);
            v12 = *(_QWORD *)(a1 + 32);
            result = v11;
            if ( v11 >= v12 )
            {
              v13 = (v12 + 1) / 2 + ((v12 + 1 + ((unsigned __int64)(v12 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v12 < v13 )
              {
                sub_CB7740(a1, v13);
                result = *(_QWORD *)(a1 + 40);
              }
            }
            v14 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = result + 1;
            *(_QWORD *)(v14 + 8 * result) = (v11 - a2) | 0x80000000LL;
            if ( !*(_DWORD *)(a1 + 16) )
            {
              v15 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8 * a2);
              result = (*(_QWORD *)(a1 + 40) - a2) | *v15 & 0xF8000000LL;
              *v15 = result;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v16 = *(_QWORD *)(a1 + 40);
                v17 = *(_QWORD *)(a1 + 32);
                if ( v16 >= v17 )
                {
                  v18 = (v17 + 1) / 2 + ((v17 + 1 + ((unsigned __int64)(v17 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v17 < v18 )
                  {
                    sub_CB7740(a1, v18);
                    v16 = *(_QWORD *)(a1 + 40);
                  }
                }
                v19 = *(_QWORD *)(a1 + 24);
                *(_QWORD *)(a1 + 40) = v16 + 1;
                *(_QWORD *)(v19 + 8 * v16) = 2281701376LL;
                result = *(_QWORD *)(a1 + 40) - 1LL;
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  v20 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8 * result);
                  result = *v20 & 0xF8000000LL | 1;
                  *v20 = result;
                  if ( !*(_DWORD *)(a1 + 16) )
                  {
                    result = *(_QWORD *)(a1 + 40);
                    v21 = *(_QWORD *)(a1 + 32);
                    if ( result >= v21 )
                    {
                      v22 = (v21 + 1) / 2 + ((v21 + 1 + ((unsigned __int64)(v21 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                      if ( v21 < v22 )
                      {
                        sub_CB7740(a1, v22);
                        result = *(_QWORD *)(a1 + 40);
                      }
                    }
                    v23 = *(_QWORD *)(a1 + 24);
                    *(_QWORD *)(a1 + 40) = result + 1;
                    *(_QWORD *)(v23 + 8 * result) = 2415919106LL;
                  }
                }
              }
            }
          }
          return result;
        case 9:
          return result;
        case 10:
          v42 = v5;
          sub_CB7820((_QWORD *)a1, 2013265920, v5 - a2 + 1, a2);
          v24 = v42;
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v25 = *(_QWORD *)(a1 + 40);
            v26 = *(_QWORD *)(a1 + 32);
            v27 = v25;
            if ( v25 >= v26 )
            {
              v28 = (v26 + 1) / 2 + ((v26 + 1 + ((unsigned __int64)(v26 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v26 < v28 )
              {
                sub_CB7740(a1, v28);
                v27 = *(_QWORD *)(a1 + 40);
                v24 = v42;
              }
            }
            v29 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = v27 + 1;
            *(_QWORD *)(v29 + 8 * v27) = (v25 - a2) | 0x80000000LL;
            if ( !*(_DWORD *)(a1 + 16) )
            {
              *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * a2) = (*(_QWORD *)(a1 + 40) - a2)
                                                         | *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * a2) & 0xF8000000LL;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v30 = *(_QWORD *)(a1 + 40);
                v31 = *(_QWORD *)(a1 + 32);
                if ( v30 >= v31 )
                {
                  v32 = (v31 + 1) / 2 + ((v31 + 1 + ((unsigned __int64)(v31 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v31 < v32 )
                  {
                    v43 = v24;
                    sub_CB7740(a1, v32);
                    v30 = *(_QWORD *)(a1 + 40);
                    v24 = v43;
                  }
                }
                v33 = *(_QWORD *)(a1 + 24);
                *(_QWORD *)(a1 + 40) = v30 + 1;
                *(_QWORD *)(v33 + 8 * v30) = 2281701376LL;
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * (*(_QWORD *)(a1 + 40) - 1LL)) = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * (*(_QWORD *)(a1 + 40) - 1LL))
                                                                                       & 0xF8000000LL
                                                                                       | 1;
                  if ( !*(_DWORD *)(a1 + 16) )
                  {
                    v34 = *(_QWORD *)(a1 + 40);
                    v35 = *(_QWORD *)(a1 + 32);
                    if ( v34 >= v35 )
                    {
                      v36 = (v35 + 1) / 2 + ((v35 + 1 + ((unsigned __int64)(v35 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                      if ( v35 < v36 )
                      {
                        v44 = v24;
                        sub_CB7740(a1, v36);
                        v34 = *(_QWORD *)(a1 + 40);
                        v24 = v44;
                      }
                    }
                    v37 = *(_QWORD *)(a1 + 24);
                    *(_QWORD *)(a1 + 40) = v34 + 1;
                    *(_QWORD *)(v37 + 8 * v34) = 2415919106LL;
                  }
                }
              }
            }
          }
          --a4;
          result = sub_CB77B0((_QWORD *)a1, a2 + 1, v24 + 1);
          v5 = *(_QWORD *)(a1 + 40);
          a2 = result;
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v9 = 8;
            a3 = 1;
            continue;
          }
          return result;
        case 11:
          sub_CB7820((_QWORD *)a1, 1207959552, v5 - a2 + 1, a2);
          result = *(unsigned int *)(a1 + 16);
          if ( !(_DWORD)result )
          {
            v38 = *(_QWORD *)(a1 + 40);
            v39 = *(_QWORD *)(a1 + 32);
            result = v38;
            if ( v38 >= v39 )
            {
              v40 = (v39 + 1) / 2 + ((v39 + 1 + ((unsigned __int64)(v39 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v39 < v40 )
              {
                sub_CB7740(a1, v40);
                result = *(_QWORD *)(a1 + 40);
              }
            }
            v41 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = result + 1;
            *(_QWORD *)(v41 + 8 * result) = (v38 - a2) | 0x50000000;
          }
          return result;
        case 18:
          --a4;
          --a3;
          a2 = sub_CB77B0((_QWORD *)a1, a2, v5);
          goto LABEL_47;
        case 19:
          --a3;
          a2 = sub_CB77B0((_QWORD *)a1, a2, v5);
LABEL_47:
          result = *(unsigned int *)(a1 + 16);
          v5 = *(_QWORD *)(a1 + 40);
          if ( (_DWORD)result )
            return result;
          goto LABEL_2;
        default:
          result = (__int64)byte_4F85140;
          *(_DWORD *)(a1 + 16) = 15;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
          return result;
      }
    }
  }
  return result;
}
