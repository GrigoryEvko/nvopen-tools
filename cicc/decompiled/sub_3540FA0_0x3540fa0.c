// Function: sub_3540FA0
// Address: 0x3540fa0
//
void __fastcall sub_3540FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r15
  __int64 *v7; // rbx
  __int64 *v8; // r13
  unsigned __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rax
  _QWORD *v19; // r14
  _QWORD *v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rcx
  __int64 v26; // r14
  __int64 *v27; // r15
  __int64 *v28; // r12
  __int64 v29; // r13
  char v30; // al
  __int64 v31; // [rsp+0h] [rbp-140h]
  __int64 *v32; // [rsp+20h] [rbp-120h]
  __int64 v33; // [rsp+38h] [rbp-108h]
  __int64 v34; // [rsp+40h] [rbp-100h]
  unsigned __int64 v35; // [rsp+40h] [rbp-100h]
  unsigned __int64 v36; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v37; // [rsp+48h] [rbp-F8h]
  __int64 *v38; // [rsp+48h] [rbp-F8h]
  _QWORD v39[2]; // [rsp+50h] [rbp-F0h] BYREF
  _BYTE *v40; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v41; // [rsp+68h] [rbp-D8h]
  _BYTE v42[32]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 *v43; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v44; // [rsp+98h] [rbp-A8h]
  _BYTE v45[32]; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 *v46; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v47; // [rsp+C8h] [rbp-78h]
  _BYTE v48[112]; // [rsp+D0h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 + 48);
  v33 = *(_QWORD *)(a2 + 56);
  if ( v6 != v33 )
  {
LABEL_4:
    while ( (unsigned __int16)(*(_WORD *)(*(_QWORD *)v6 + 68LL) - 19) > 1u )
    {
LABEL_3:
      v6 += 256;
      if ( v33 == v6 )
        return;
    }
    v40 = v42;
    v41 = 0x400000000LL;
    v43 = (__int64 *)v45;
    v44 = 0x400000000LL;
    v7 = *(__int64 **)(v6 + 40);
    v8 = &v7[2 * *(unsigned int *)(v6 + 48)];
    if ( v7 == v8 )
      goto LABEL_22;
    while ( 1 )
    {
      v9 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = (*v7 >> 1) & 3;
      v11 = *(_QWORD *)v9;
      if ( v10 != 1 )
        break;
      if ( *(_WORD *)(v11 + 68) != 68 && *(_WORD *)(v11 + 68) )
      {
LABEL_11:
        v7 += 2;
        if ( v8 == v7 )
          goto LABEL_18;
      }
      else
      {
        v12 = (unsigned int)v41;
        v13 = (unsigned int)v41 + 1LL;
        if ( v13 > HIDWORD(v41) )
        {
          sub_C8D5F0((__int64)&v40, v42, v13, 8u, a5, a6);
          v12 = (unsigned int)v41;
        }
        v7 += 2;
        *(_QWORD *)&v40[8 * v12] = v9;
        LODWORD(v41) = v41 + 1;
        if ( v8 == v7 )
        {
LABEL_18:
          if ( !(_DWORD)v41 || !(_DWORD)v44 )
          {
LABEL_20:
            v14 = (unsigned __int64)v43;
            if ( v43 != (__int64 *)v45 )
              goto LABEL_21;
            goto LABEL_22;
          }
          v46 = (__int64 *)v48;
          v47 = 0x800000000LL;
          v34 = v6;
          v17 = 0;
          while ( 2 )
          {
            v18 = *(_QWORD *)&v40[8 * v17];
            v19 = *(_QWORD **)(v18 + 120);
            v20 = &v19[2 * *(unsigned int *)(v18 + 128)];
            if ( v20 == v19 )
            {
LABEL_41:
              if ( ++v17 < (unsigned __int64)(unsigned int)v41 )
                continue;
              v6 = v34;
              if ( !(_DWORD)v47 )
              {
                if ( v46 != (__int64 *)v48 )
                  _libc_free((unsigned __int64)v46);
                goto LABEL_20;
              }
              v31 = v34;
              v38 = v46;
              v32 = &v46[(unsigned int)v47];
              v24 = a2 + 3528;
              do
              {
                v25 = (__int64)v43;
                v26 = *v38;
                v27 = v43;
                v28 = &v43[(unsigned int)v44];
                v35 = *v38 & 0xFFFFFFFFFFFFFFF9LL;
                if ( v28 != v43 )
                {
                  do
                  {
                    v29 = *v27;
                    v30 = sub_2F90B20(v24, v26, *v27, v25, a5, a6);
                    if ( v26 != v29 && v30 != 1 )
                    {
                      v39[1] = 3;
                      v39[0] = v35 | 6;
                      sub_2F8F1B0(v29, (__int64)v39, 1u, v25, a5, a6);
                      sub_2F90A20(v24, v29, v26);
                    }
                    ++v27;
                  }
                  while ( v28 != v27 );
                }
                ++v38;
              }
              while ( v32 != v38 );
              v6 = v31;
              if ( v46 != (__int64 *)v48 )
                _libc_free((unsigned __int64)v46);
              v14 = (unsigned __int64)v43;
              if ( v43 != (__int64 *)v45 )
LABEL_21:
                _libc_free(v14);
LABEL_22:
              if ( v40 == v42 )
                goto LABEL_3;
              _libc_free((unsigned __int64)v40);
              v6 += 256;
              if ( v33 == v6 )
                return;
              goto LABEL_4;
            }
            break;
          }
          while ( 1 )
          {
LABEL_35:
            if ( (*v19 & 6) != 0 )
              goto LABEL_34;
            v23 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
            if ( *(_WORD *)(*(_QWORD *)v23 + 68LL) == 68
              || *(_WORD *)(*(_QWORD *)v23 + 68LL) == 0
              || *(_WORD *)(*(_QWORD *)v23 + 68LL) == 19 )
            {
              break;
            }
            v22 = (unsigned int)v47;
            a5 = (unsigned int)v47 + 1LL;
            if ( a5 > HIDWORD(v47) )
            {
              v37 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
              sub_C8D5F0((__int64)&v46, v48, (unsigned int)v47 + 1LL, 8u, a5, a6);
              v22 = (unsigned int)v47;
              v23 = v37;
            }
            v19 += 2;
            v46[v22] = v23;
            LODWORD(v47) = v47 + 1;
            if ( v20 == v19 )
              goto LABEL_41;
          }
          v21 = (unsigned int)v41;
          a5 = (unsigned int)v41 + 1LL;
          if ( a5 > HIDWORD(v41) )
          {
            v36 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
            sub_C8D5F0((__int64)&v40, v42, (unsigned int)v41 + 1LL, 8u, a5, a6);
            v21 = (unsigned int)v41;
            v23 = v36;
          }
          *(_QWORD *)&v40[8 * v21] = v23;
          LODWORD(v41) = v41 + 1;
LABEL_34:
          v19 += 2;
          if ( v20 == v19 )
            goto LABEL_41;
          goto LABEL_35;
        }
      }
    }
    if ( !v10 && *(_WORD *)(v11 + 68) && *(_WORD *)(v11 + 68) != 68 && *(_DWORD *)(v9 + 208) )
    {
      v15 = (unsigned int)v44;
      v16 = (unsigned int)v44 + 1LL;
      if ( v16 > HIDWORD(v44) )
      {
        sub_C8D5F0((__int64)&v43, v45, v16, 8u, a5, a6);
        v15 = (unsigned int)v44;
      }
      v43[v15] = v9;
      LODWORD(v44) = v44 + 1;
    }
    goto LABEL_11;
  }
}
