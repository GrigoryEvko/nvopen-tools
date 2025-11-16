// Function: sub_F9E3A0
// Address: 0xf9e3a0
//
__int64 __fastcall sub_F9E3A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  _QWORD *v3; // rax
  __int64 v5; // rdx
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 i; // r13
  __int64 v15; // rsi
  __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rcx
  bool v21; // r8
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // r10
  __int64 v27; // rdx
  __int64 v28; // rdx
  int v29; // eax
  unsigned int v30; // edi
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rsi
  char v37; // dh
  char v38; // al
  __int64 v39; // rdx
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 j; // rbx
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int8 *v47; // rcx
  __int64 v48; // r14
  int v49; // esi
  __int64 v50; // rdx
  __int64 v51; // rbx
  const char *v52; // r12
  unsigned __int64 v53; // rax
  __int64 v54; // [rsp+10h] [rbp-90h]
  __int64 v55; // [rsp+10h] [rbp-90h]
  __int64 v56; // [rsp+18h] [rbp-88h]
  __int64 v57; // [rsp+18h] [rbp-88h]
  __int64 v58; // [rsp+20h] [rbp-80h]
  __int64 v59; // [rsp+20h] [rbp-80h]
  __int64 v60; // [rsp+28h] [rbp-78h]
  bool v62; // [rsp+3Eh] [rbp-62h]
  unsigned __int8 v63; // [rsp+3Fh] [rbp-61h]
  __m128i v64; // [rsp+40h] [rbp-60h] BYREF
  __int64 v65; // [rsp+50h] [rbp-50h] BYREF
  __int64 v66; // [rsp+58h] [rbp-48h]
  __int64 v67; // [rsp+60h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( v2 != v3[5] )
    return 0;
  v5 = v3[2];
  if ( !v5 || *(_QWORD *)(v5 + 8) )
    return 0;
  v7 = v3[4];
  if ( v7 == v2 + 48 )
    v7 = 0;
  v63 = sub_F8F640(v7, a1 + 24);
  if ( v63 )
  {
    v60 = 0;
    if ( (*(_BYTE *)(v8 + 2) & 1) != 0 )
    {
      v11 = *(_QWORD *)(v8 + 32 * (1 - v10));
      v60 = v11;
      if ( v11 )
      {
        v65 = sub_AA5930(v11);
        v12 = v65;
        for ( i = v13; v65 != i; v12 = v65 )
        {
          v15 = *(_QWORD *)(v12 - 8);
          v16 = 0x1FFFFFFFE0LL;
          v17 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
          if ( v17 )
          {
            v18 = 0;
            do
            {
              if ( v2 == *(_QWORD *)(v15 + 32LL * *(unsigned int *)(v12 + 72) + 8 * v18) )
              {
                v16 = 32 * v18;
                goto LABEL_17;
              }
              ++v18;
            }
            while ( v17 != (_DWORD)v18 );
            v16 = 0x1FFFFFFFE0LL;
          }
LABEL_17:
          v19 = *(_QWORD *)(v15 + v16);
          v20 = 0;
          v21 = 0;
          if ( *(_BYTE *)v19 == 84 )
          {
            v20 = v19;
            v21 = *(_QWORD *)(v19 + 40) == v2;
          }
          v22 = *(_QWORD *)(v2 + 16);
          if ( v22 )
          {
            while ( 1 )
            {
              v23 = *(_QWORD *)(v22 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
                break;
              v22 = *(_QWORD *)(v22 + 8);
              if ( !v22 )
                goto LABEL_42;
            }
LABEL_21:
            v24 = *(_QWORD *)(v23 + 40);
            v25 = v19;
            if ( v21 )
            {
              v26 = *(_QWORD *)(v20 - 8);
              v27 = 0x1FFFFFFFE0LL;
              if ( (*(_DWORD *)(v20 + 4) & 0x7FFFFFF) != 0 )
              {
                v28 = 0;
                do
                {
                  if ( v24 == *(_QWORD *)(v26 + 32LL * *(unsigned int *)(v20 + 72) + 8 * v28) )
                  {
                    v27 = 32 * v28;
                    goto LABEL_27;
                  }
                  ++v28;
                }
                while ( (*(_DWORD *)(v20 + 4) & 0x7FFFFFF) != (_DWORD)v28 );
                v27 = 0x1FFFFFFFE0LL;
              }
LABEL_27:
              v25 = *(_QWORD *)(v26 + v27);
            }
            if ( v17 == *(_DWORD *)(v12 + 72) )
            {
              v62 = v21;
              v55 = v20;
              v57 = v24;
              v59 = v25;
              sub_B48D90(v12);
              v25 = v59;
              v21 = v62;
              v20 = v55;
              v24 = v57;
              v17 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
            }
            v29 = (v17 + 1) & 0x7FFFFFF;
            v30 = v29 | *(_DWORD *)(v12 + 4) & 0xF8000000;
            v31 = *(_QWORD *)(v12 - 8) + 32LL * (unsigned int)(v29 - 1);
            *(_DWORD *)(v12 + 4) = v30;
            if ( *(_QWORD *)v31 )
            {
              v32 = *(_QWORD *)(v31 + 8);
              **(_QWORD **)(v31 + 16) = v32;
              if ( v32 )
                *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 16);
            }
            *(_QWORD *)v31 = v25;
            if ( v25 )
            {
              v33 = *(_QWORD *)(v25 + 16);
              *(_QWORD *)(v31 + 8) = v33;
              if ( v33 )
                *(_QWORD *)(v33 + 16) = v31 + 8;
              *(_QWORD *)(v31 + 16) = v25 + 16;
              *(_QWORD *)(v25 + 16) = v31;
            }
            *(_QWORD *)(*(_QWORD *)(v12 - 8)
                      + 32LL * *(unsigned int *)(v12 + 72)
                      + 8LL * ((*(_DWORD *)(v12 + 4) & 0x7FFFFFFu) - 1)) = v24;
            while ( 1 )
            {
              v22 = *(_QWORD *)(v22 + 8);
              if ( !v22 )
                break;
              v23 = *(_QWORD *)(v22 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
              {
                v17 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
                goto LABEL_21;
              }
            }
          }
LABEL_42:
          sub_F8F2F0((__int64)&v65);
        }
        v34 = sub_AA4FF0(v60);
        v35 = 1;
        v36 = v34;
        v56 = v34;
        v38 = v37;
        if ( !v36 )
          v38 = 0;
        BYTE1(v35) = v38;
        v54 = v35;
        v40 = sub_AA5930(v2);
        v65 = v40;
        if ( v39 != v40 )
        {
          v58 = v39;
          do
          {
            LOBYTE(v66) = 1;
            sub_F8F2F0((__int64)&v65);
            if ( *(_QWORD *)(v40 + 16) && (unsigned __int8)sub_B463C0(v40, v2) )
            {
              v41 = sub_F92F30(v60);
              v64.m128i_i64[0] = v41;
              for ( j = v42; j != v64.m128i_i64[0]; v41 = v64.m128i_i64[0] )
              {
                v44 = *(_QWORD *)(*(_QWORD *)(v41 + 24) + 40LL);
                if ( v2 != v44 )
                  sub_F0A850(v40, v40, v44);
                v64.m128i_i64[0] = *(_QWORD *)(v64.m128i_i64[0] + 8);
                sub_D4B000(v64.m128i_i64);
              }
              sub_B444E0((_QWORD *)v40, v56, v54);
              v45 = sub_ACADE0(*(__int64 ***)(v40 + 8));
              sub_F0A850(v40, v45, v2);
            }
            v40 = v65;
          }
          while ( v58 != v65 );
        }
      }
    }
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v46 = *(_QWORD *)(v2 + 16);
    while ( v46 )
    {
      v47 = *(unsigned __int8 **)(v46 + 24);
      v48 = v46;
      v46 = *(_QWORD *)(v46 + 8);
      v49 = *v47;
      v50 = (unsigned int)(v49 - 30);
      if ( (unsigned __int8)(v49 - 30) <= 0xAu )
      {
        v51 = v48;
        do
        {
          v51 = *(_QWORD *)(v51 + 8);
          if ( !v51 )
            goto LABEL_63;
LABEL_62:
          ;
        }
        while ( (unsigned __int8)(**(_BYTE **)(v51 + 24) - 30) > 0xAu );
        while ( 1 )
        {
LABEL_63:
          v52 = (const char *)*((_QWORD *)v47 + 5);
          if ( v60 )
          {
            sub_AA5980(v2, *((_QWORD *)v47 + 5), 0);
            v53 = sub_986580((__int64)v52);
            sub_BD2ED0(v53, v2, v60);
            if ( a2 )
            {
              v64.m128i_i64[0] = (__int64)v52;
              v64.m128i_i64[1] = v60 & 0xFFFFFFFFFFFFFFFBLL;
              sub_F9E360((__int64)&v65, &v64);
              v64.m128i_i64[0] = (__int64)v52;
              v64.m128i_i64[1] = v2 | 4;
              sub_F9E360((__int64)&v65, &v64);
            }
          }
          else
          {
            if ( a2 )
            {
              sub_FFB3D0(a2, v65, (v66 - v65) >> 4);
              if ( v65 != v66 )
                v66 = v65;
            }
            sub_F56CD0(v52, a2, v50, (__int64)v47, v9, v10);
          }
          if ( !v51 )
            goto LABEL_73;
          v47 = *(unsigned __int8 **)(v51 + 24);
          v51 = *(_QWORD *)(v51 + 8);
          if ( v51 )
            goto LABEL_62;
        }
      }
    }
LABEL_73:
    if ( a2 )
      sub_FFB3D0(a2, v65, (v66 - v65) >> 4);
    sub_F34560(v2, a2, 0);
    if ( v65 )
      j_j___libc_free_0(v65, v67 - v65);
  }
  else
  {
    return 0;
  }
  return v63;
}
