// Function: sub_1F81F00
// Address: 0x1f81f00
//
__int64 __fastcall sub_1F81F00(__int64 *a1, __int64 a2)
{
  __int16 v3; // ax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r9d
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // r12
  unsigned __int8 v18; // r13
  char v19; // al
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  int v26; // r9d
  _QWORD *v27; // r12
  __int64 (__fastcall **v28)(); // rax
  __int64 v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rdi
  char v33; // r8
  __int64 v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // [rsp+0h] [rbp-B0h]
  __int64 v37; // [rsp+8h] [rbp-A8h]
  __int64 v38; // [rsp+18h] [rbp-98h]
  char v39; // [rsp+26h] [rbp-8Ah]
  unsigned __int8 v40; // [rsp+27h] [rbp-89h]
  unsigned int v41; // [rsp+28h] [rbp-88h]
  unsigned int v42; // [rsp+3Ch] [rbp-74h] BYREF
  __int64 v43; // [rsp+40h] [rbp-70h] BYREF
  __int64 v44; // [rsp+48h] [rbp-68h]
  __int128 v45; // [rsp+50h] [rbp-60h] BYREF
  __int64 (__fastcall **v46)(); // [rsp+60h] [rbp-50h] BYREF
  __int64 v47; // [rsp+68h] [rbp-48h]
  __int64 v48; // [rsp+70h] [rbp-40h]
  __int64 *v49; // [rsp+78h] [rbp-38h]

  v3 = *(_WORD *)(a2 + 24);
  if ( v3 == 185 )
  {
    if ( (*(_WORD *)(a2 + 26) & 0x380) != 0 )
      return 0;
    v4 = *(unsigned __int8 *)(a2 + 88);
    if ( !(_BYTE)v4 )
      return 0;
    v5 = a1[1] + 5 * v4;
    if ( ((*(_BYTE *)(v5 + 71886) >> 4) & 0xB) != 0 && ((*(_BYTE *)(v5 + 71887) >> 4) & 0xB) != 0 )
      return 0;
    v39 = 1;
    v38 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
  }
  else
  {
    if ( v3 != 186 )
      return 0;
    if ( (*(_WORD *)(a2 + 26) & 0x380) != 0 )
      return 0;
    v21 = *(unsigned __int8 *)(a2 + 88);
    if ( !(_BYTE)v21 )
      return 0;
    v22 = a1[1] + 5 * v21;
    if ( (*(_BYTE *)(v22 + 71886) & 0xB) != 0 && (*(_BYTE *)(v22 + 71887) & 0xB) != 0 )
      return 0;
    v39 = 0;
    v38 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL);
  }
  v6 = *(_QWORD *)(v38 + 48);
  if ( !v6 || !*(_QWORD *)(v6 + 32) )
    return 0;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v6 + 16);
    if ( a2 != v7 && (unsigned int)*(unsigned __int16 *)(v7 + 24) - 52 <= 1 )
    {
      v8 = a1[1];
      v43 = 0;
      LODWORD(v44) = 0;
      *(_QWORD *)&v45 = 0;
      DWORD2(v45) = 0;
      v42 = 0;
      v9 = *(__int64 (**)())(*(_QWORD *)v8 + 1008LL);
      if ( v9 != sub_1F6BB50 )
      {
        v40 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *, __int128 *, unsigned int *, __int64))v9)(
                v8,
                a2,
                v7,
                &v43,
                &v45,
                &v42,
                *a1);
        if ( v40 )
        {
          if ( !sub_1D185B0(v45) )
          {
            v13 = v43;
            v14 = *(unsigned __int16 *)(v43 + 24);
            if ( (unsigned __int16)v14 > 0x24u || (v10 = 0x1000004100LL, !_bittest64(&v10, v14)) )
            {
              if ( *(_QWORD *)(v43 + 48) )
              {
                v37 = v6;
                v15 = *(_QWORD *)(v43 + 48);
                v36 = v7;
                do
                {
                  v16 = *(_QWORD *)(v15 + 16);
                  if ( v38 != v16 && (unsigned int)*(unsigned __int16 *)(v16 + 24) - 52 <= 1 )
                  {
                    v17 = *(_QWORD *)(v16 + 48);
                    if ( !v17 )
                      goto LABEL_27;
                    v18 = 0;
                    do
                    {
                      v19 = sub_1F6D430(v16, *(_QWORD *)(v17 + 16), *a1, a1[1]);
                      v17 = *(_QWORD *)(v17 + 32);
                      if ( !v19 )
                        v18 = v40;
                    }
                    while ( v17 );
                    if ( !v18 )
                    {
LABEL_27:
                      v6 = v37;
                      goto LABEL_9;
                    }
                  }
                  v15 = *(_QWORD *)(v15 + 32);
                }
                while ( v15 );
                v7 = v36;
                v6 = v37;
              }
              if ( !(unsigned __int8)sub_1D19270(a2, v7, v13, v10, v11, v12)
                && !(unsigned __int8)sub_1D19270(v7, a2, v23, v24, v25, v26) )
              {
                break;
              }
            }
          }
        }
      }
    }
LABEL_9:
    v6 = *(_QWORD *)(v6 + 32);
    if ( !v6 )
      return 0;
  }
  v27 = (_QWORD *)*a1;
  v41 = v42;
  v28 = *(__int64 (__fastcall ***)())(a2 + 72);
  v46 = v28;
  if ( v39 )
  {
    if ( v28 )
      sub_1F6CA20((__int64 *)&v46);
    LODWORD(v47) = *(_DWORD *)(a2 + 64);
    v29 = sub_1D26680(v27, a2, 0, (__int64)&v46, v43, v44, v45, v41);
    sub_17CD270((__int64 *)&v46);
    v30 = *(_QWORD *)(*a1 + 664);
    v48 = *a1;
    v47 = v30;
    *(_QWORD *)(v48 + 664) = &v46;
    v31 = *a1;
    v46 = off_49FFF30;
    v49 = a1;
    sub_1D44C70(v31, a2, 0, v29, 0);
    sub_1D44C70(*a1, a2, 1, v29, 2u);
    sub_1F81E80(a1, a2);
    v32 = *a1;
    v33 = 1;
  }
  else
  {
    if ( v28 )
      sub_1F6CA20((__int64 *)&v46);
    LODWORD(v47) = *(_DWORD *)(a2 + 64);
    v29 = (__int64)sub_1D25500(v27, a2, 0, (__int64)&v46, v43, v44, v45, v41);
    sub_17CD270((__int64 *)&v46);
    v34 = *(_QWORD *)(*a1 + 664);
    v48 = *a1;
    v47 = v34;
    *(_QWORD *)(v48 + 664) = &v46;
    v35 = *a1;
    v46 = off_49FFF30;
    v49 = a1;
    sub_1D44C70(v35, a2, 0, v29, 1u);
    sub_1F81E80(a1, a2);
    v32 = *a1;
    v33 = 0;
  }
  sub_1D44C70(v32, v7, 0, v29, v33 & 1);
  sub_1F81E80(a1, v7);
  *(_QWORD *)(v48 + 664) = v47;
  return v40;
}
