// Function: sub_2D45870
// Address: 0x2d45870
//
__int64 *__fastcall sub_2D45870(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        int a6,
        char a7,
        __int64 *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 *v11; // rbx
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 v19; // r15
  __int16 v20; // dx
  _QWORD *v21; // rax
  __int64 v22; // r14
  unsigned int *v23; // r15
  __int64 v24; // r12
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 *result; // rax
  __int64 v30; // r14
  __int64 v31; // rdx
  int v32; // r13d
  unsigned int *v33; // r14
  __int64 v34; // r13
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rdx
  int v38; // r15d
  __int64 v39; // r15
  __int64 v40; // rbx
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rdx
  int v44; // r13d
  __int64 v45; // r12
  unsigned int *v46; // rbx
  __int64 v47; // rdx
  unsigned int v48; // esi
  char v49; // [rsp+0h] [rbp-C0h]
  __int16 v50; // [rsp+Ch] [rbp-B4h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  _DWORD v56[8]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v57; // [rsp+50h] [rbp-70h]
  _QWORD v58[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v59; // [rsp+80h] [rbp-40h]

  v11 = a1;
  v52 = *(_QWORD *)(a4 + 8);
  v12 = *(_BYTE *)(v52 + 8);
  if ( v12 <= 3u || v12 == 5 || (v12 & 0xFD) == 4 || (unsigned __int8)(v12 - 17) <= 1u )
  {
    v13 = sub_BCAE30(v52);
    v58[1] = v14;
    v58[0] = v13;
    v15 = sub_CA1930(v58);
    v16 = sub_BCD140((_QWORD *)a1[9], v15);
    v57 = 257;
    v17 = v16;
    if ( v16 == *(_QWORD *)(a4 + 8) )
    {
      v18 = a4;
    }
    else
    {
      v18 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(
              a1[10],
              49,
              a4,
              v16);
      if ( !v18 )
      {
        v59 = 257;
        v18 = sub_B51D30(49, a4, v17, (__int64)v58, 0, 0);
        if ( (unsigned __int8)sub_920620(v18) )
        {
          v37 = a1[12];
          v38 = *((_DWORD *)a1 + 26);
          if ( v37 )
            sub_B99FD0(v18, 3u, v37);
          sub_B45150(v18, v38);
        }
        (*(void (__fastcall **)(__int64, __int64, _DWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
          a1[11],
          v18,
          v56,
          a1[7],
          a1[8]);
        if ( *a1 != *a1 + 16LL * *((unsigned int *)a1 + 2) )
        {
          v39 = *a1 + 16LL * *((unsigned int *)a1 + 2);
          v40 = *a1;
          do
          {
            v41 = *(_QWORD *)(v40 + 8);
            v42 = *(_DWORD *)v40;
            v40 += 16;
            sub_B99FD0(v18, v42, v41);
          }
          while ( v39 != v40 );
          v11 = a1;
        }
      }
    }
    v57 = 257;
    if ( v17 == *(_QWORD *)(a3 + 8) )
    {
      v19 = a3;
    }
    else
    {
      v19 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v11[10] + 120LL))(
              v11[10],
              49,
              a3,
              v17);
      if ( !v19 )
      {
        v59 = 257;
        v19 = sub_B51D30(49, a3, v17, (__int64)v58, 0, 0);
        if ( (unsigned __int8)sub_920620(v19) )
        {
          v31 = v11[12];
          v32 = *((_DWORD *)v11 + 26);
          if ( v31 )
            sub_B99FD0(v19, 3u, v31);
          sub_B45150(v19, v32);
        }
        (*(void (__fastcall **)(__int64, __int64, _DWORD *, __int64, __int64))(*(_QWORD *)v11[11] + 16LL))(
          v11[11],
          v19,
          v56,
          v11[7],
          v11[8]);
        v33 = (unsigned int *)*v11;
        v34 = *v11 + 16LL * *((unsigned int *)v11 + 2);
        if ( *v11 != v34 )
        {
          do
          {
            v35 = *((_QWORD *)v33 + 1);
            v36 = *v33;
            v33 += 4;
            sub_B99FD0(v19, v36, v35);
          }
          while ( (unsigned int *)v34 != v33 );
        }
      }
    }
    v49 = 1;
  }
  else
  {
    v49 = 0;
    v19 = a3;
    v18 = a4;
  }
  switch ( a6 )
  {
    case 2:
    case 5:
      v20 = 2;
      break;
    case 4:
    case 6:
      v20 = 4;
      break;
    case 7:
      v20 = a6;
      break;
    default:
      BUG();
  }
  v50 = v20;
  v59 = 257;
  v21 = sub_BD2C40(80, unk_3F148C4);
  v22 = (__int64)v21;
  if ( v21 )
    sub_B4D5A0((__int64)v21, a2, v19, v18, a5, a6, v50, a7, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v11[11] + 16LL))(
    v11[11],
    v22,
    v58,
    v11[7],
    v11[8]);
  v23 = (unsigned int *)*v11;
  v24 = *v11 + 16LL * *((unsigned int *)v11 + 2);
  if ( *v11 != v24 )
  {
    do
    {
      v25 = *((_QWORD *)v23 + 1);
      v26 = *v23;
      v23 += 4;
      sub_B99FD0(v22, v26, v25);
    }
    while ( (unsigned int *)v24 != v23 );
  }
  if ( a10 )
    sub_2D42CA0(v22, a10);
  v58[0] = "success";
  v59 = 259;
  v56[0] = 1;
  v27 = sub_94D3D0((unsigned int **)v11, v22, (__int64)v56, 1, (__int64)v58);
  HIBYTE(v59) = 1;
  *a8 = v27;
  v58[0] = "newloaded";
  LOBYTE(v59) = 3;
  v56[0] = 0;
  v28 = sub_94D3D0((unsigned int **)v11, v22, (__int64)v56, 1, (__int64)v58);
  result = a9;
  *a9 = v28;
  if ( v49 )
  {
    v57 = 257;
    if ( v52 == *(_QWORD *)(v28 + 8) )
    {
      v30 = v28;
    }
    else
    {
      v30 = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v11[10] + 120LL))(v11[10], 49, v28);
      if ( !v30 )
      {
        v59 = 257;
        v30 = sub_B51D30(49, v28, v52, (__int64)v58, 0, 0);
        if ( (unsigned __int8)sub_920620(v30) )
        {
          v43 = v11[12];
          v44 = *((_DWORD *)v11 + 26);
          if ( v43 )
            sub_B99FD0(v30, 3u, v43);
          sub_B45150(v30, v44);
        }
        (*(void (__fastcall **)(__int64, __int64, _DWORD *, __int64, __int64))(*(_QWORD *)v11[11] + 16LL))(
          v11[11],
          v30,
          v56,
          v11[7],
          v11[8]);
        v45 = *v11 + 16LL * *((unsigned int *)v11 + 2);
        if ( *v11 != v45 )
        {
          v46 = (unsigned int *)*v11;
          do
          {
            v47 = *((_QWORD *)v46 + 1);
            v48 = *v46;
            v46 += 4;
            sub_B99FD0(v30, v48, v47);
          }
          while ( (unsigned int *)v45 != v46 );
        }
      }
    }
    *a9 = v30;
    return a9;
  }
  return result;
}
