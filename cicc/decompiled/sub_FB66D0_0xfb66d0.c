// Function: sub_FB66D0
// Address: 0xfb66d0
//
__int64 __fastcall sub_FB66D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int64 v3; // rax
  int v4; // edx
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // r13
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 *v21; // r13
  __int64 *v22; // rbx
  __int64 *v23; // rdx
  __int64 *v24; // rdi
  __int64 v25; // r12
  __int64 v26; // rbx
  int i; // ecx
  __int64 v28; // rax
  __int64 v29; // rax
  const char **v30; // rcx
  __int64 v31; // r15
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  int v34; // edx
  _QWORD *v35; // rdi
  _QWORD *v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // [rsp+18h] [rbp-E8h]
  __int64 *v41; // [rsp+18h] [rbp-E8h]
  __int64 v43; // [rsp+30h] [rbp-D0h]
  __int64 *v45; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v46; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v47; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-A8h]
  char v49; // [rsp+70h] [rbp-90h]
  __int64 v50; // [rsp+80h] [rbp-80h] BYREF
  __int64 v51; // [rsp+88h] [rbp-78h]
  __int64 v52; // [rsp+90h] [rbp-70h]
  __int64 v53; // [rsp+98h] [rbp-68h]
  __int64 *v54; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v55; // [rsp+A8h] [rbp-58h]
  _BYTE v56[80]; // [rsp+B0h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == v2 + 48 )
  {
    v5 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    v4 = *(unsigned __int8 *)(v3 - 24);
    v5 = 0;
    v6 = v3 - 24;
    if ( (unsigned int)(v4 - 30) < 0xB )
      v5 = v6;
  }
  v7 = sub_AA4FF0(v2);
  v8 = v5 + 24;
  LODWORD(v9) = sub_F8F640(v7, v5 + 24);
  if ( (_BYTE)v9 )
  {
    v50 = 0;
    v54 = (__int64 *)v56;
    v55 = 0x400000000LL;
    v51 = 0;
    v11 = *(_QWORD *)(a2 - 32);
    v52 = 0;
    v53 = 0;
    if ( (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != 0 )
    {
      v12 = 0;
      v13 = v11;
      v14 = 8LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
      do
      {
        v46 = *(_QWORD *)(*(_QWORD *)(v13 - 8) + 32LL * *(unsigned int *)(v13 + 72) + v12);
        v15 = *(_QWORD *)(*(_QWORD *)(v13 - 8) + 4 * v12);
        if ( v2 == sub_AA5780(v46) )
        {
          v18 = sub_AA4FF0(v46);
          if ( !v18 )
            BUG();
          v19 = 0;
          if ( *(_BYTE *)(v18 - 24) == 95 )
            v19 = v18 - 24;
          if ( v15 == v19 )
          {
            v8 = *(_QWORD *)(v46 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v8 == v46 + 48 )
              goto LABEL_61;
            if ( !v8 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
            {
LABEL_61:
              v20 = *(_QWORD *)(v15 + 32);
              if ( v20 == *(_QWORD *)(v15 + 40) + 48LL )
                v20 = 0;
              v8 = 0;
            }
            else
            {
              v20 = *(_QWORD *)(v15 + 32);
              if ( v20 == *(_QWORD *)(v15 + 40) + 48LL )
                v20 = 0;
            }
            if ( (unsigned __int8)sub_F8F640(v20, v8) )
            {
              if ( (_DWORD)v52 )
              {
                v8 = (unsigned __int64)&v50;
                sub_D6CB10((__int64)&v47, (__int64)&v50, (__int64 *)&v46);
                if ( v49 )
                {
                  v8 = v46;
                  sub_B1A4E0((__int64)&v54, v46);
                }
              }
              else
              {
                v8 = (unsigned __int64)&v54[(unsigned int)v55];
                if ( (_QWORD *)v8 == sub_F8ED40(v54, v8, (__int64 *)&v46) )
                {
                  v8 = v16;
                  sub_B1A4E0((__int64)&v54, v16);
                  if ( (unsigned int)v55 > 4 )
                  {
                    v40 = v12;
                    v21 = &v54[(unsigned int)v55];
                    v22 = v54;
                    do
                    {
                      v23 = v22;
                      v8 = (unsigned __int64)&v50;
                      ++v22;
                      sub_D6CB10((__int64)&v47, (__int64)&v50, v23);
                    }
                    while ( v21 != v22 );
                    v12 = v40;
                  }
                }
              }
            }
          }
        }
        v12 += 8;
      }
      while ( v14 != v12 );
      v9 = v13;
      v24 = v54;
      v25 = a1;
      if ( (_DWORD)v55 )
      {
        v45 = v54;
        v41 = &v54[(unsigned int)v55];
        do
        {
          v26 = *v45;
          for ( i = *(_DWORD *)(v9 + 4) & 0x7FFFFFF; i; i = *(_DWORD *)(v9 + 4) & 0x7FFFFFF )
          {
            v28 = 0;
            while ( v26 != *(_QWORD *)(*(_QWORD *)(v9 - 8) + 32LL * *(unsigned int *)(v9 + 72) + 8 * v28) )
            {
              if ( i == (_DWORD)++v28 )
                goto LABEL_34;
            }
            sub_AA5980(v2, v26, 1u);
          }
LABEL_34:
          v29 = *(_QWORD *)(v26 + 16);
          while ( v29 )
          {
            v30 = *(const char ***)(v29 + 24);
            v31 = v29;
            v29 = *(_QWORD *)(v29 + 8);
            v32 = (unsigned int)*(unsigned __int8 *)v30 - 30;
            if ( (unsigned __int8)(*(_BYTE *)v30 - 30) <= 0xAu )
            {
              while ( 1 )
              {
                v31 = *(_QWORD *)(v31 + 8);
                if ( !v31 )
                  break;
                while ( (unsigned __int8)(**(_BYTE **)(v31 + 24) - 30) <= 0xAu )
                {
                  sub_F56CD0(v30[5], *(_QWORD *)(v25 + 8), v32, (__int64)v30, v16, v17);
                  v30 = *(const char ***)(v31 + 24);
                  v31 = *(_QWORD *)(v31 + 8);
                  if ( !v31 )
                    goto LABEL_40;
                }
              }
LABEL_40:
              sub_F56CD0(v30[5], *(_QWORD *)(v25 + 8), v32, (__int64)v30, v16, v17);
              break;
            }
          }
          v33 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v33 == v26 + 48 )
          {
            v35 = 0;
          }
          else
          {
            if ( !v33 )
              BUG();
            v34 = *(unsigned __int8 *)(v33 - 24);
            v35 = 0;
            v36 = (_QWORD *)(v33 - 24);
            if ( (unsigned int)(v34 - 30) < 0xB )
              v35 = v36;
          }
          sub_B43D60(v35);
          v43 = sub_BD5C60(a2);
          sub_B43C20((__int64)&v47, v26);
          v8 = unk_3F148B8;
          v37 = sub_BD2C40(72, unk_3F148B8);
          if ( v37 )
          {
            v8 = v43;
            sub_B4C8A0((__int64)v37, v43, v47, v48);
          }
          v38 = *(_QWORD *)(v25 + 8);
          if ( v38 )
          {
            v8 = (unsigned __int64)&v47;
            v47 = v26;
            v48 = v2 | 4;
            sub_FFB3D0(v38, &v47, 1);
          }
          ++v45;
        }
        while ( v41 != v45 );
        v39 = *(_QWORD *)(v2 + 16);
        if ( v39 )
        {
          while ( (unsigned __int8)(**(_BYTE **)(v39 + 24) - 30) > 0xAu )
          {
            v39 = *(_QWORD *)(v39 + 8);
            if ( !v39 )
              goto LABEL_57;
          }
        }
        else
        {
LABEL_57:
          v8 = *(_QWORD *)(v25 + 8);
          sub_F34560(v2, v8, 0);
        }
        v24 = v54;
        LOBYTE(v9) = (_DWORD)v55 != 0;
      }
      else
      {
        LODWORD(v9) = 0;
      }
      if ( v24 != (__int64 *)v56 )
        _libc_free(v24, v8);
    }
    else
    {
      LODWORD(v9) = 0;
    }
    sub_C7D6A0(v51, 8LL * (unsigned int)v53, 8);
  }
  return (unsigned int)v9;
}
