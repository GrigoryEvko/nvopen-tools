// Function: sub_25941E0
// Address: 0x25941e0
//
__int64 __fastcall sub_25941E0(__int64 *a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  _BYTE *v9; // rax
  int v10; // edx
  unsigned int v11; // r14d
  int v12; // eax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  int v17; // ebx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rcx
  unsigned __int64 v22; // rbx
  unsigned __int8 *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // r10
  __int64 v27; // r9
  __m128i v28; // rax
  unsigned __int8 *v29; // rax
  __int64 v30; // rdx
  int v31; // edx
  int v32; // edx
  __int64 v33; // [rsp+0h] [rbp-B0h]
  __int64 v34; // [rsp+8h] [rbp-A8h]
  _BYTE *v35; // [rsp+18h] [rbp-98h]
  char v36; // [rsp+2Dh] [rbp-83h] BYREF
  char v37; // [rsp+2Eh] [rbp-82h] BYREF
  char v38; // [rsp+2Fh] [rbp-81h] BYREF
  __m128i v39; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int8 *v40; // [rsp+40h] [rbp-70h]
  __int64 v41; // [rsp+48h] [rbp-68h]
  __m128i v42[6]; // [rsp+50h] [rbp-60h] BYREF

  if ( !(unsigned __int8)sub_B19060(*a1 + 200, (__int64)a2, a3, a4)
    && !(unsigned __int8)sub_B19060(*a1 + 104, (__int64)a2, v6, v7) )
  {
    v9 = (_BYTE *)*((_QWORD *)a2 - 4);
    v35 = v9;
    if ( v9 )
    {
      if ( !*v9 )
      {
        v10 = *a2;
        v11 = 0;
        v12 = v10 - 29;
        if ( v10 != 40 )
          goto LABEL_7;
LABEL_28:
        v13 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
        if ( (a2[7] & 0x80u) != 0 )
        {
          while ( 1 )
          {
            v14 = sub_BD2BC0((__int64)a2);
            v16 = v14 + v15;
            if ( (a2[7] & 0x80u) == 0 )
              break;
            if ( !(unsigned int)((v16 - sub_BD2BC0((__int64)a2)) >> 4) )
              goto LABEL_29;
            if ( (a2[7] & 0x80u) == 0 )
              goto LABEL_37;
            v17 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
            if ( (a2[7] & 0x80u) == 0 )
              BUG();
            v18 = sub_BD2BC0((__int64)a2);
            v20 = 32LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
LABEL_16:
            v21 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
            if ( v11 >= (unsigned int)((v21 - 32 - v13 - v20) >> 5) || (unsigned __int64)v11 >= *((_QWORD *)v35 + 13) )
              return 1;
            v22 = *(_QWORD *)&a2[32 * (v11 - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
            if ( v22 )
            {
              v23 = &a2[-v21];
              if ( (a2[7] & 0x40) != 0 )
                v23 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
              v42[0] = (__m128i)((unsigned __int64)&v23[32 * v11] | 3);
              nullsub_1518();
              v24 = a1[1];
              v25 = *a1;
              v39 = _mm_loadu_si128(v42);
              sub_257DDD0(v24, v25, &v39, 2, &v36, 0, 0);
              if ( v36 )
              {
                v26 = a1[1];
                v27 = *a1;
                v37 = 0;
                v33 = v26;
                v34 = v27;
                v28.m128i_i64[0] = sub_250D2C0(v22, 0);
                v42[0] = v28;
                v29 = (unsigned __int8 *)sub_2527850(v33, v42, v34, &v37, 2u);
                v40 = v29;
                v41 = v30;
                if ( !v37 )
                {
                  if ( !(_BYTE)v30 )
                    goto LABEL_26;
                  if ( !v29 )
                    return 1;
                  v31 = *v29;
                  if ( (unsigned int)(v31 - 12) <= 1
                    || *(_BYTE *)(*(_QWORD *)(v22 + 8) + 8LL) == 14
                    && (_BYTE)v31 == 20
                    && (sub_258F340((_QWORD *)a1[1], *a1, &v39, 2, &v38, 0, 0), v38) )
                  {
LABEL_26:
                    sub_BED950((__int64)v42, *a1 + 104, (__int64)a2);
                  }
                }
              }
            }
            v32 = *a2;
            ++v11;
            v12 = v32 - 29;
            if ( v32 == 40 )
              goto LABEL_28;
LABEL_7:
            v13 = 0;
            if ( v12 != 56 )
            {
              if ( v12 != 5 )
                BUG();
              v13 = 64;
            }
            if ( (a2[7] & 0x80u) == 0 )
              goto LABEL_29;
          }
          if ( (unsigned int)(v16 >> 4) )
LABEL_37:
            BUG();
        }
LABEL_29:
        v20 = 0;
        goto LABEL_16;
      }
    }
  }
  return 1;
}
