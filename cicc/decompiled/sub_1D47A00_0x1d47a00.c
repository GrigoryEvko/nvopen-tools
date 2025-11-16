// Function: sub_1D47A00
// Address: 0x1d47a00
//
__int64 __fastcall sub_1D47A00(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned int v6; // r12d
  int v7; // r8d
  int v8; // r9d
  __int64 *v10; // rax
  __int64 v11; // r12
  __int64 v12; // r11
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // r15
  char v16; // dl
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 *v19; // rax
  __int64 *v20; // rsi
  __int64 *v21; // rcx
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // r14
  char v25; // dl
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 *v28; // rax
  __int64 *v29; // rsi
  __int64 *v30; // rcx
  __int64 *v31; // rsi
  __int64 *v32; // rcx
  __int64 v33; // [rsp+8h] [rbp-198h]
  _BYTE *v35; // [rsp+30h] [rbp-170h] BYREF
  __int64 v36; // [rsp+38h] [rbp-168h]
  _BYTE v37[128]; // [rsp+40h] [rbp-160h] BYREF
  __int64 v38; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 *v39; // [rsp+C8h] [rbp-D8h]
  __int64 *v40; // [rsp+D0h] [rbp-D0h]
  __int64 v41; // [rsp+D8h] [rbp-C8h]
  int v42; // [rsp+E0h] [rbp-C0h]
  _BYTE v43[184]; // [rsp+E8h] [rbp-B8h] BYREF

  v4 = a2;
  v5 = a3;
  v6 = 0;
  v39 = (__int64 *)v43;
  v40 = (__int64 *)v43;
  v35 = v37;
  v38 = 0;
  v41 = 16;
  v42 = 0;
  v36 = 0x1000000000LL;
  if ( !(unsigned __int8)sub_1D18C70(a3, a2) )
  {
    v10 = v39;
    if ( v40 != v39 )
      goto LABEL_8;
    v31 = &v39[HIDWORD(v41)];
    if ( v39 != v31 )
    {
      v32 = 0;
      while ( v5 != *v10 )
      {
        if ( *v10 == -2 )
          v32 = v10;
        if ( v31 == ++v10 )
        {
          if ( !v32 )
            goto LABEL_63;
          *v32 = v5;
          --v42;
          ++v38;
          goto LABEL_9;
        }
      }
      goto LABEL_9;
    }
LABEL_63:
    if ( HIDWORD(v41) < (unsigned int)v41 )
    {
      ++HIDWORD(v41);
      *v31 = v5;
      ++v38;
    }
    else
    {
LABEL_8:
      sub_16CCBA0((__int64)&v38, v5);
    }
LABEL_9:
    v11 = *(_QWORD *)(v5 + 32);
    v12 = v11 + 40LL * *(unsigned int *)(v5 + 56);
    if ( v11 != v12 )
    {
      v33 = v5;
      v13 = *(_QWORD *)(v5 + 32);
      v14 = v4;
      v15 = v12;
      while ( 1 )
      {
        v18 = *(_QWORD *)v13;
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v13 + 40LL) + 16LL * *(unsigned int *)(v13 + 8)) != 1 || !a4)
          && v14 != v18 )
        {
          v19 = v39;
          if ( v40 == v39 )
          {
            v20 = &v39[HIDWORD(v41)];
            if ( v39 != v20 )
            {
              v21 = 0;
              while ( v18 != *v19 )
              {
                if ( *v19 == -2 )
                  v21 = v19;
                if ( v20 == ++v19 )
                {
                  if ( !v21 )
                    goto LABEL_57;
                  *v21 = v18;
                  v17 = (unsigned int)v36;
                  --v42;
                  ++v38;
                  if ( (unsigned int)v36 < HIDWORD(v36) )
                    goto LABEL_13;
                  goto LABEL_27;
                }
              }
              goto LABEL_14;
            }
LABEL_57:
            if ( HIDWORD(v41) < (unsigned int)v41 )
            {
              ++HIDWORD(v41);
              *v20 = v18;
              ++v38;
LABEL_12:
              v17 = (unsigned int)v36;
              if ( (unsigned int)v36 >= HIDWORD(v36) )
              {
LABEL_27:
                sub_16CD150((__int64)&v35, v37, 0, 8, v7, v8);
                v17 = (unsigned int)v36;
              }
LABEL_13:
              *(_QWORD *)&v35[8 * v17] = v18;
              LODWORD(v36) = v36 + 1;
              goto LABEL_14;
            }
          }
          sub_16CCBA0((__int64)&v38, *(_QWORD *)v13);
          if ( v16 )
            goto LABEL_12;
        }
LABEL_14:
        v13 += 40;
        if ( v15 == v13 )
        {
          v4 = v14;
          v5 = v33;
          break;
        }
      }
    }
    if ( v5 == a1 || (v22 = *(_QWORD *)(a1 + 32), v22 == v22 + 40LL * *(unsigned int *)(a1 + 56)) )
    {
LABEL_49:
      v6 = sub_1D15B50(v4, (__int64)&v38, (__int64)&v35, 0, 1, v8);
      goto LABEL_2;
    }
    v23 = v22 + 40LL * *(unsigned int *)(a1 + 56);
    v24 = *(_QWORD *)(a1 + 32);
    while ( 1 )
    {
      v27 = *(_QWORD *)v24;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v24 + 40LL) + 16LL * *(unsigned int *)(v24 + 8)) != 1 || !a4) && v4 != v27 )
      {
        v28 = v39;
        if ( v40 == v39 )
        {
          v29 = &v39[HIDWORD(v41)];
          if ( v39 != v29 )
          {
            v30 = 0;
            while ( v27 != *v28 )
            {
              if ( *v28 == -2 )
                v30 = v28;
              if ( v29 == ++v28 )
              {
                if ( !v30 )
                  goto LABEL_59;
                *v30 = v27;
                --v42;
                ++v38;
                goto LABEL_33;
              }
            }
            goto LABEL_36;
          }
LABEL_59:
          if ( HIDWORD(v41) < (unsigned int)v41 )
          {
            ++HIDWORD(v41);
            *v29 = v27;
            ++v38;
LABEL_33:
            v26 = (unsigned int)v36;
            if ( (unsigned int)v36 >= HIDWORD(v36) )
            {
              sub_16CD150((__int64)&v35, v37, 0, 8, v7, v8);
              v26 = (unsigned int)v36;
            }
            *(_QWORD *)&v35[8 * v26] = v27;
            LODWORD(v36) = v36 + 1;
            goto LABEL_36;
          }
        }
        sub_16CCBA0((__int64)&v38, *(_QWORD *)v24);
        if ( v25 )
          goto LABEL_33;
      }
LABEL_36:
      v24 += 40;
      if ( v23 == v24 )
        goto LABEL_49;
    }
  }
LABEL_2:
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  if ( v40 != v39 )
    _libc_free((unsigned __int64)v40);
  return v6;
}
