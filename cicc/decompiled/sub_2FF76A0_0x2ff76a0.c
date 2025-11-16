// Function: sub_2FF76A0
// Address: 0x2ff76a0
//
__int64 __fastcall sub_2FF76A0(
        __int64 a1,
        int a2,
        unsigned __int16 *a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  int *v10; // r8
  unsigned __int64 v11; // rdi
  int *v12; // r14
  int v13; // eax
  __int64 v14; // rdx
  unsigned int *v15; // rbx
  unsigned int v16; // r10d
  unsigned int *v17; // rax
  char v18; // dl
  __int16 v19; // bx
  unsigned __int16 *v20; // rax
  unsigned __int16 *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned int *v26; // r14
  bool v27; // r9
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // r15
  unsigned __int64 v32; // rdx
  int *v33; // [rsp+0h] [rbp-140h]
  __int64 v37; // [rsp+28h] [rbp-118h]
  char v38; // [rsp+30h] [rbp-110h]
  int *v39; // [rsp+30h] [rbp-110h]
  unsigned int v40; // [rsp+30h] [rbp-110h]
  int *v41; // [rsp+38h] [rbp-108h]
  unsigned int v42; // [rsp+4Ch] [rbp-F4h] BYREF
  unsigned int *v43; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v44; // [rsp+58h] [rbp-E8h]
  _BYTE v45[128]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v46; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v47; // [rsp+E8h] [rbp-58h] BYREF
  unsigned __int64 v48; // [rsp+F0h] [rbp-50h]
  __int64 *v49; // [rsp+F8h] [rbp-48h]
  __int64 *v50; // [rsp+100h] [rbp-40h]
  __int64 v51; // [rsp+108h] [rbp-38h]

  v7 = a2 & 0x7FFFFFFF;
  v8 = *(_QWORD *)(a6 + 32);
  v37 = v8;
  if ( *(_DWORD *)(v8 + 248) > (unsigned int)v7 )
  {
    v9 = *(_QWORD *)(v8 + 240) + 40 * v7;
    if ( v9 )
    {
      LODWORD(v47) = 0;
      v48 = 0;
      v49 = &v47;
      v50 = &v47;
      v51 = 0;
      v44 = 0x2000000000LL;
      v10 = *(int **)(v9 + 8);
      v43 = (unsigned int *)v45;
      v41 = &v10[*(unsigned int *)(v9 + 16)];
      if ( v41 != v10 )
      {
        if ( !*(_DWORD *)v9 )
        {
          v13 = *v10;
          v42 = *v10;
          if ( a7 && v13 < 0 )
          {
            v12 = v10;
            v14 = 0;
            goto LABEL_34;
          }
          v12 = v10;
          goto LABEL_9;
        }
        while ( 1 )
        {
          while ( 1 )
          {
LABEL_5:
            v11 = v48;
            v12 = v10 + 1;
            if ( v10 + 1 == v41 )
              goto LABEL_30;
            v13 = v10[1];
            v14 = v51;
            v42 = v13;
            if ( a7 && v13 < 0 )
LABEL_34:
              v42 = *(_DWORD *)(*(_QWORD *)(a7 + 32) + 4LL * (v13 & 0x7FFFFFFF));
            if ( !v14 )
              break;
            sub_2DCBF00((__int64)&v46, &v42);
            v10 = v12;
            if ( v18 )
              goto LABEL_16;
          }
LABEL_9:
          v15 = &v43[(unsigned int)v44];
          if ( v43 == v15 )
          {
            if ( (unsigned int)v44 <= 0x1FuLL )
            {
              v16 = v42;
LABEL_46:
              v32 = (unsigned int)v44 + 1LL;
              if ( v32 > HIDWORD(v44) )
              {
                v40 = v16;
                sub_C8D5F0((__int64)&v43, v45, v32, 4u, (__int64)v43, a6);
                v16 = v40;
                v15 = &v43[(unsigned int)v44];
              }
              *v15 = v16;
              LODWORD(v44) = v44 + 1;
              goto LABEL_49;
            }
          }
          else
          {
            v16 = v42;
            v17 = v43;
            while ( *v17 != v42 )
            {
              if ( v15 == ++v17 )
                goto LABEL_35;
            }
            if ( v15 != v17 )
            {
              v10 = v12;
              goto LABEL_5;
            }
LABEL_35:
            if ( (unsigned int)v44 <= 0x1FuLL )
              goto LABEL_46;
            v33 = v12;
            v26 = v43;
            do
            {
              v29 = sub_2DCC990(&v46, (__int64)&v47, v26);
              v31 = (_QWORD *)v30;
              if ( v30 )
              {
                v27 = v29 || (__int64 *)v30 == &v47 || *v26 < *(_DWORD *)(v30 + 32);
                v38 = v27;
                v28 = sub_22077B0(0x28u);
                *(_DWORD *)(v28 + 32) = *v26;
                sub_220F040(v38, v28, v31, &v47);
                ++v51;
              }
              ++v26;
            }
            while ( v15 != v26 );
            v12 = v33;
          }
          LODWORD(v44) = 0;
          sub_2DCBF00((__int64)&v46, &v42);
LABEL_49:
          v10 = v12;
LABEL_16:
          v19 = v42;
          if ( v42 - 1 <= 0x3FFFFFFE && (*(_QWORD *)(*(_QWORD *)(v37 + 384) + 8LL * (v42 >> 6)) & (1LL << v42)) == 0 )
          {
            v20 = a3;
            v21 = &a3[a4];
            v22 = (2 * a4) >> 3;
            v23 = (2 * a4) >> 1;
            if ( v22 > 0 )
            {
              while ( v42 != *v20 )
              {
                if ( v42 == v20[1] )
                {
                  if ( v21 == v20 + 1 )
                    goto LABEL_5;
                  goto LABEL_26;
                }
                if ( v42 == v20[2] )
                {
                  if ( v21 == v20 + 2 )
                    goto LABEL_5;
                  goto LABEL_26;
                }
                if ( v42 == v20[3] )
                {
                  if ( v21 == v20 + 3 )
                    goto LABEL_5;
                  goto LABEL_26;
                }
                v20 += 4;
                if ( &a3[4 * v22] == v20 )
                {
                  v23 = v21 - v20;
                  goto LABEL_56;
                }
              }
              goto LABEL_25;
            }
LABEL_56:
            switch ( v23 )
            {
              case 2LL:
                goto LABEL_68;
              case 3LL:
                if ( v42 == *v20 )
                  goto LABEL_25;
                ++v20;
LABEL_68:
                if ( v42 != *v20 )
                {
                  ++v20;
                  goto LABEL_70;
                }
LABEL_25:
                if ( v21 != v20 )
                  goto LABEL_26;
                break;
              case 1LL:
LABEL_70:
                if ( v42 != *v20 )
                  v20 = &a3[a4];
                if ( v21 != v20 )
                {
LABEL_26:
                  v24 = a5[1];
                  if ( (unsigned __int64)(v24 + 1) > a5[2] )
                  {
                    v39 = v10;
                    sub_C8D290((__int64)a5, a5 + 3, v24 + 1, 2u, (__int64)v10, a6);
                    v24 = a5[1];
                    v10 = v39;
                  }
                  *(_WORD *)(*a5 + 2 * v24) = v19;
                  ++a5[1];
                }
                break;
            }
          }
        }
      }
      v11 = 0;
LABEL_30:
      sub_2FF57E0(v11);
      if ( v43 != (unsigned int *)v45 )
        _libc_free((unsigned __int64)v43);
    }
  }
  return 0;
}
