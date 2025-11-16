// Function: sub_2F47200
// Address: 0x2f47200
//
void __fastcall sub_2F47200(__int64 a1, __int64 a2)
{
  unsigned int *v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned int *i; // r9
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned int v13; // ecx
  unsigned int *v14; // rax
  unsigned int *v15; // rdx
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // r13
  __int64 v19; // r12
  unsigned int v20; // r10d
  unsigned int *v21; // rbx
  unsigned int *v22; // rax
  unsigned int *v23; // r12
  char v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 *v29; // r8
  __int64 v30; // [rsp+8h] [rbp-E8h]
  __int64 v31; // [rsp+10h] [rbp-E0h]
  __int64 v32; // [rsp+38h] [rbp-B8h]
  _QWORD *v34; // [rsp+50h] [rbp-A0h]
  unsigned int v35; // [rsp+50h] [rbp-A0h]
  __int64 v36; // [rsp+58h] [rbp-98h]
  unsigned int v37; // [rsp+6Ch] [rbp-84h] BYREF
  unsigned int *v38; // [rsp+70h] [rbp-80h] BYREF
  __int64 v39; // [rsp+78h] [rbp-78h]
  char v40[8]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v41; // [rsp+88h] [rbp-68h] BYREF
  __int64 v42; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v43; // [rsp+98h] [rbp-58h]
  __int64 *v44; // [rsp+A0h] [rbp-50h]
  __int64 *v45; // [rsp+A8h] [rbp-48h]
  __int64 v46; // [rsp+B0h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 424) )
  {
    v2 = *(unsigned int **)(a2 + 192);
    v3 = a1;
    for ( i = (unsigned int *)sub_2E33140(a2); v2 != i; sub_2F42240(a1, *i, 2) )
      ;
    LODWORD(v42) = 0;
    v38 = (unsigned int *)v40;
    v39 = 0x200000000LL;
    v44 = &v42;
    v45 = &v42;
    v43 = 0;
    v7 = *(_QWORD *)(a2 + 56);
    v46 = 0;
    v36 = a2 + 48;
    if ( v7 != a2 + 48 )
    {
      while ( 1 )
      {
        while ( (unsigned __int16)(*(_WORD *)(v7 + 68) - 4) <= 2u )
        {
          if ( (*(_BYTE *)v7 & 4) == 0 && (*(_BYTE *)(v7 + 44) & 8) != 0 )
          {
            do
              v7 = *(_QWORD *)(v7 + 8);
            while ( (*(_BYTE *)(v7 + 44) & 8) != 0 );
          }
LABEL_10:
          v7 = *(_QWORD *)(v7 + 8);
          if ( v7 == v36 )
            goto LABEL_11;
        }
        v16 = *(_QWORD *)(a1 + 24);
        v17 = *(__int64 (**)())(*(_QWORD *)v16 + 1336LL);
        if ( v17 == sub_2E2F9B0
          || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64, unsigned int *))v17)(
                v16,
                v7,
                0,
                v4,
                v5,
                i) )
        {
          goto LABEL_11;
        }
        v5 = *(_QWORD *)(v7 + 32);
        v18 = v5;
        v19 = v5 + 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
        if ( v5 == v19 )
          goto LABEL_53;
        v32 = v7;
        do
        {
          while ( *(_BYTE *)v18 )
          {
LABEL_32:
            v18 += 40;
            if ( v19 == v18 )
              goto LABEL_52;
          }
          v20 = *(_DWORD *)(v18 + 8);
          v37 = v20;
          if ( v46 )
            goto LABEL_51;
          v4 = (unsigned int)v39;
          v21 = &v38[(unsigned int)v39];
          if ( v38 != v21 )
          {
            v22 = v38;
            while ( v20 != *v22 )
            {
              if ( v21 == ++v22 )
                goto LABEL_40;
            }
            if ( v21 != v22 )
              goto LABEL_32;
LABEL_40:
            if ( (unsigned int)v39 > 1uLL )
            {
              v31 = v19;
              v23 = v38;
              v30 = v18;
              do
              {
                v26 = sub_2DCC990(&v41, (__int64)&v42, v23);
                if ( v27 )
                {
                  v24 = v26 || (__int64 *)v27 == &v42 || *v23 < *(_DWORD *)(v27 + 32);
                  v34 = (_QWORD *)v27;
                  v25 = sub_22077B0(0x28u);
                  *(_DWORD *)(v25 + 32) = *v23;
                  sub_220F040(v24, v25, v34, &v42);
                  ++v46;
                }
                ++v23;
              }
              while ( v21 != v23 );
              v19 = v31;
              v18 = v30;
              goto LABEL_50;
            }
LABEL_69:
            if ( (unsigned __int64)(unsigned int)v39 + 1 > HIDWORD(v39) )
            {
              v35 = v20;
              sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 4u, v5, (__int64)i);
              v20 = v35;
              v21 = &v38[(unsigned int)v39];
            }
            *v21 = v20;
            LODWORD(v39) = v39 + 1;
            goto LABEL_32;
          }
          if ( (unsigned int)v39 <= 1uLL )
            goto LABEL_69;
LABEL_50:
          LODWORD(v39) = 0;
LABEL_51:
          v18 += 40;
          sub_2DCBE50((__int64)&v41, &v37);
        }
        while ( v19 != v18 );
LABEL_52:
        v7 = v32;
LABEL_53:
        if ( (*(_BYTE *)v7 & 4) != 0 )
          goto LABEL_10;
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
          v7 = *(_QWORD *)(v7 + 8);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v7 == v36 )
        {
LABEL_11:
          v3 = a1;
          break;
        }
      }
    }
    v8 = *(_QWORD *)(v3 + 416);
    if ( v8 + 24LL * *(unsigned int *)(v3 + 424) == v8 )
    {
LABEL_26:
      *(_DWORD *)(v3 + 424) = 0;
      sub_2F42310(v43);
      if ( v38 != (unsigned int *)v40 )
        _libc_free((unsigned __int64)v38);
      return;
    }
    v9 = v3;
    v10 = v8 + 24LL * *(unsigned int *)(v3 + 424);
    v11 = v8;
    while ( 1 )
    {
      v12 = *(unsigned __int16 *)(v11 + 12);
      if ( !(_WORD)v12 )
        goto LABEL_24;
      if ( *(_BYTE *)(v11 + 16) )
        goto LABEL_24;
      v13 = (unsigned __int16)v12;
      if ( *(_DWORD *)(*(_QWORD *)(v9 + 808)
                     + 4LL * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v9 + 16) + 8LL) + 24 * v12 + 16) & 0xFFF)) == 2 )
        goto LABEL_24;
      if ( v46 )
      {
        v28 = v43;
        if ( !v43 )
          goto LABEL_61;
        v29 = &v42;
        do
        {
          if ( v13 > *(_DWORD *)(v28 + 32) )
          {
            v28 = *(_QWORD *)(v28 + 24);
          }
          else
          {
            v29 = (__int64 *)v28;
            v28 = *(_QWORD *)(v28 + 16);
          }
        }
        while ( v28 );
        if ( v29 == &v42 || v13 < *((_DWORD *)v29 + 8) )
          goto LABEL_61;
      }
      else
      {
        v14 = v38;
        v15 = &v38[(unsigned int)v39];
        if ( v38 == v15 )
          goto LABEL_61;
        while ( v13 != *v14 )
        {
          if ( v15 == ++v14 )
            goto LABEL_61;
        }
        if ( v15 == v14 )
        {
LABEL_61:
          sub_2F41820(v9, v7, *(_DWORD *)(v11 + 8), v13);
          goto LABEL_24;
        }
      }
      sub_2F41820(v9, *(_QWORD *)(a2 + 56), *(_DWORD *)(v11 + 8), v13);
LABEL_24:
      v11 += 24;
      if ( v11 == v10 )
      {
        v3 = v9;
        goto LABEL_26;
      }
    }
  }
}
