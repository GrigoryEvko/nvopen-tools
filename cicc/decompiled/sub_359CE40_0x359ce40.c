// Function: sub_359CE40
// Address: 0x359ce40
//
__int64 __fastcall sub_359CE40(__int64 a1, __int64 a2, int a3, _QWORD *a4, _QWORD *a5, _QWORD *a6)
{
  __int64 v7; // rbx
  __int64 v8; // r13
  int v9; // r8d
  __int64 result; // rax
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // r8
  __int64 v14; // r9
  __int32 v15; // eax
  __int64 v16; // rdi
  __int32 v17; // r13d
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  unsigned __int8 v27; // dl
  int v28; // r13d
  __int64 v29; // r12
  __int64 v30; // rdi
  __int64 v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rbx
  __int64 v34; // r14
  int v35; // ecx
  __int64 v36; // rsi
  int v37; // r9d
  unsigned int i; // eax
  __int64 v39; // rdx
  char v41; // [rsp+Fh] [rbp-E1h]
  __int32 v44; // [rsp+1Ch] [rbp-D4h]
  __int64 v45; // [rsp+20h] [rbp-D0h]
  __int64 v46; // [rsp+20h] [rbp-D0h]
  __int64 v48; // [rsp+30h] [rbp-C0h]
  __int64 v49; // [rsp+40h] [rbp-B0h]
  __int64 v50; // [rsp+48h] [rbp-A8h]
  int v51; // [rsp+54h] [rbp-9Ch] BYREF
  __int64 v52; // [rsp+58h] [rbp-98h] BYREF
  __int64 v53[2]; // [rsp+60h] [rbp-90h] BYREF
  __int64 v54[4]; // [rsp+70h] [rbp-80h] BYREF
  __m128i v55; // [rsp+90h] [rbp-60h] BYREF
  __int64 v56; // [rsp+A0h] [rbp-50h]
  __int64 v57; // [rsp+A8h] [rbp-48h]

  v7 = a1;
  v8 = *(_QWORD *)a1;
  v9 = sub_3598DB0(*(_QWORD *)a1, a2);
  result = (unsigned int)(a3 + *(_DWORD *)(v8 + 96) - *(_DWORD *)(a1 + 128));
  if ( (int)result > v9 )
  {
    v41 = 1;
  }
  else
  {
    if ( (_DWORD)result != v9 )
      return result;
    v41 = 0;
  }
  v11 = *(_QWORD *)(a2 + 32);
  result = v11 + 40LL * (unsigned int)sub_2E88FE0(a2);
  v50 = result;
  if ( result != *(_QWORD *)(a2 + 32) )
  {
    v12 = *(_QWORD *)(a2 + 32);
    v49 = 32LL * a3;
    result = (__int64)&v51;
    do
    {
      while ( 2 )
      {
        if ( !*(_BYTE *)v12 )
        {
          v27 = *(_BYTE *)(v12 + 3);
          result = (v27 & 0x40) != 0;
          if ( ((unsigned __int8)result & (v27 >> 4)) == 0 )
          {
            v28 = *(_DWORD *)(v12 + 8);
            v29 = *a5 + v49;
            v51 = v28;
            result = (__int64)sub_359BBF0(v29, &v51);
            v48 = result;
            v30 = result;
            if ( result )
            {
              result = *(_QWORD *)(v29 + 8) + 8LL * *(unsigned int *)(v29 + 24);
              if ( v30 != result )
              {
                if ( v41 )
                {
                  v14 = (unsigned int)*sub_2FFAE70(
                                         *a4
                                       + 32LL * (*(_DWORD *)(*(_QWORD *)v7 + 96LL) + a3 - *(_DWORD *)(v7 + 128) - 1),
                                         &v51);
LABEL_7:
                  v44 = v14;
                  v15 = sub_2EC06C0(
                          *(_QWORD *)(v7 + 24),
                          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 24) + 56LL) + 16LL * (v51 & 0x7FFFFFFF))
                        & 0xFFFFFFFFFFFFFFF8LL,
                          byte_3F871B3,
                          0,
                          v13,
                          v14);
                  v16 = *(_QWORD *)(v7 + 88);
                  v52 = 0;
                  v17 = v15;
                  v18 = *(_QWORD *)(v7 + 32);
                  memset(v54, 0, 24);
                  v45 = *(_QWORD *)(v18 + 8);
                  v19 = (__int64 *)sub_2E311E0(v16);
                  v53[0] = (__int64)sub_2F26260(*(_QWORD *)(v7 + 88), v19, v54, v45, v17);
                  v53[1] = v20;
                  v21 = sub_3598AB0(v53, *(_DWORD *)(v48 + 4), 0, 0);
                  v22 = *(_QWORD *)(v7 + 88);
                  v23 = v21[1];
                  v55.m128i_i8[0] = 4;
                  v57 = v22;
                  v55.m128i_i32[0] &= 0xFFF000FF;
                  v56 = 0;
                  sub_2E8EAD0(v23, *v21, &v55);
                  v24 = sub_3598AB0(v21, v44, 0, 0);
                  v25 = *(_QWORD *)(v7 + 80);
                  v26 = v24[1];
                  v55.m128i_i8[0] = 4;
                  v57 = v25;
                  v55.m128i_i32[0] &= 0xFFF000FF;
                  v56 = 0;
                  sub_2E8EAD0(v26, *v24, &v55);
                  if ( v54[0] )
                    sub_B91220((__int64)v54, v54[0]);
                  if ( v52 )
                    sub_B91220((__int64)&v52, v52);
                  result = (__int64)sub_2FFAE70(*a6 + v49, &v51);
                  *(_DWORD *)result = v17;
                }
                else
                {
                  v31 = *(_QWORD *)(v7 + 48);
                  result = sub_2E311E0(v31);
                  v32 = *(_QWORD *)(v31 + 56);
                  v55.m128i_i64[0] = v32;
                  if ( v32 != result )
                  {
                    v46 = v7;
                    v33 = v31;
                    v34 = result;
                    do
                    {
                      v35 = *(_DWORD *)(v32 + 40) & 0xFFFFFF;
                      if ( v35 == 1 )
                      {
                        if ( !v28 )
                          goto LABEL_30;
                      }
                      else
                      {
                        v36 = *(_QWORD *)(v32 + 32);
                        v37 = 0;
                        for ( i = 1; i != v35; i += 2 )
                        {
                          while ( v33 != *(_QWORD *)(v36 + 40LL * (i + 1) + 24) )
                          {
                            i += 2;
                            if ( v35 == i )
                              goto LABEL_25;
                          }
                          v39 = i;
                          v37 = *(_DWORD *)(v36 + 40 * v39 + 8);
                        }
LABEL_25:
                        if ( v28 == v37 )
                        {
LABEL_30:
                          v7 = v46;
                          v14 = (unsigned int)sub_3598140(v32, *(_QWORD *)(v46 + 48));
                          goto LABEL_7;
                        }
                      }
                      result = sub_2FD79B0(v55.m128i_i64);
                      v32 = v55.m128i_i64[0];
                    }
                    while ( v55.m128i_i64[0] != v34 );
                    v7 = v46;
                    v12 += 40;
                    if ( v50 != v12 )
                      continue;
                    return result;
                  }
                }
              }
            }
          }
        }
        break;
      }
      v12 += 40;
    }
    while ( v50 != v12 );
  }
  return result;
}
