// Function: sub_2B1C4B0
// Address: 0x2b1c4b0
//
void __fastcall sub_2B1C4B0(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v8; // r12
  unsigned int *v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r13
  unsigned int *v12; // rax
  char *v13; // r10
  char *v14; // r11
  __int64 v15; // r14
  int v16; // r9d
  unsigned int *v17; // r11
  unsigned int *v18; // rax
  unsigned int v19; // eax
  unsigned int *v21; // [rsp+10h] [rbp-B0h]
  char *v22; // [rsp+18h] [rbp-A8h]
  int srca; // [rsp+20h] [rbp-A0h]
  unsigned int *src; // [rsp+20h] [rbp-A0h]
  unsigned int *v25; // [rsp+28h] [rbp-98h]
  unsigned int *v26; // [rsp+28h] [rbp-98h]

  v25 = a1;
  if ( a4 )
  {
    v8 = a5;
    if ( a5 )
    {
      v9 = a2;
      v10 = a4;
      if ( a5 + a4 == 2 )
      {
        a6 = a1;
        v17 = a2;
LABEL_12:
        src = a6;
        v26 = v17;
        if ( sub_2B1BC20((__int64 **)&a7, *v17, *a6) )
        {
          v19 = *src;
          *src = *v26;
          *v26 = v19;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v11 = v10 / 2;
        v12 = sub_2B1C1A0(
                v9,
                a3,
                &v25[v10 / 2],
                a4,
                a5,
                (__int64)a6,
                (__int64 *)_mm_loadu_si128((const __m128i *)&a7).m128i_i64[0]);
        v13 = (char *)&v25[v10 / 2];
        v14 = (char *)v12;
        v15 = v12 - v9;
        while ( 1 )
        {
          v21 = (unsigned int *)v14;
          srca = (int)v13;
          v8 -= v15;
          v22 = sub_2B12540(v13, (char *)v9, v14);
          sub_2B1C4B0((_DWORD)v25, srca, (_DWORD)v22, v11, v15, v16, a7, a8);
          v10 -= v11;
          if ( !v10 )
            break;
          a6 = (unsigned int *)v22;
          v17 = v21;
          if ( !v8 )
            break;
          if ( v8 + v10 == 2 )
            goto LABEL_12;
          v25 = (unsigned int *)v22;
          v9 = v21;
          if ( v10 > v8 )
            goto LABEL_5;
LABEL_10:
          v15 = v8 / 2;
          v18 = sub_2B1C220(
                  v25,
                  (__int64)v9,
                  &v9[v8 / 2],
                  a4,
                  a5,
                  (__int64)a6,
                  (__int64 *)_mm_loadu_si128((const __m128i *)&a7).m128i_i64[0]);
          v14 = (char *)&v9[v8 / 2];
          v13 = (char *)v18;
          v11 = v18 - v25;
        }
      }
    }
  }
}
