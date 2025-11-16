// Function: sub_23DDC70
// Address: 0x23ddc70
//
__int64 __fastcall sub_23DDC70(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v5; // r12
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 *v9; // r12
  __int64 v10; // r15
  unsigned __int8 v11; // al
  unsigned int v12; // r15d
  _QWORD *v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rdx
  unsigned int v18; // esi
  unsigned __int64 v19; // rdx
  __int16 v20; // cx
  __int64 v21; // r9
  _BYTE *v22; // rax
  __int64 v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 *v28; // rax
  __int64 *v29; // r12
  __int64 v30; // r15
  unsigned __int8 v31; // al
  unsigned int v32; // r15d
  unsigned __int8 v33; // r13
  _QWORD *v34; // rax
  __int64 v35; // r13
  __int64 v36; // r12
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // r12
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  unsigned __int8 v44; // [rsp+Fh] [rbp-A1h]
  _QWORD v47[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v48; // [rsp+40h] [rbp-70h]
  _BYTE v49[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v50; // [rsp+70h] [rbp-40h]

  v5 = *(_QWORD *)(a3 + 16);
  v6 = (_QWORD *)a2[9];
  v47[0] = "MyAlloca";
  v48 = 259;
  if ( a4 )
  {
    v7 = sub_BCB2E0(v6);
    v8 = sub_ACD640(v7, v5, 0);
    v9 = (__int64 *)sub_BCB2B0((_QWORD *)a2[9]);
    v10 = sub_AA4E30(a2[6]);
    v11 = sub_AE5260(v10, (__int64)v9);
    v12 = *(_DWORD *)(v10 + 4);
    v44 = v11;
    v50 = 257;
    v13 = sub_BD2C40(80, unk_3F10A14);
    v14 = (__int64)v13;
    if ( v13 )
      sub_B4CCA0((__int64)v13, v9, v12, v8, v44, (__int64)v49, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
      a2[11],
      v14,
      v47,
      a2[7],
      a2[8]);
    v15 = *a2;
    v16 = *a2 + 16LL * *((unsigned int *)a2 + 2);
    if ( *a2 != v16 )
    {
      do
      {
        v17 = *(_QWORD *)(v15 + 8);
        v18 = *(_DWORD *)v15;
        v15 += 16;
        sub_B99FD0(v14, v18, v17);
      }
      while ( v16 != v15 );
    }
  }
  else
  {
    v28 = (__int64 *)sub_BCB2B0(v6);
    v29 = sub_BCD420(v28, v5);
    v30 = sub_AA4E30(a2[6]);
    v31 = sub_AE5260(v30, (__int64)v29);
    v32 = *(_DWORD *)(v30 + 4);
    v33 = v31;
    v50 = 257;
    v34 = sub_BD2C40(80, unk_3F10A14);
    v14 = (__int64)v34;
    if ( v34 )
      sub_B4CCA0((__int64)v34, v29, v32, 0, v33, (__int64)v49, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
      a2[11],
      v14,
      v47,
      a2[7],
      a2[8]);
    v35 = *a2;
    v36 = *a2 + 16LL * *((unsigned int *)a2 + 2);
    if ( *a2 != v36 )
    {
      do
      {
        v37 = *(_QWORD *)(v35 + 8);
        v38 = *(_DWORD *)v35;
        v35 += 16;
        sub_B99FD0(v14, v38, v37);
      }
      while ( v36 != v35 );
    }
  }
  v19 = (unsigned int)qword_4FE10C8;
  if ( (unsigned __int64)(unsigned int)qword_4FE10C8 < *(_QWORD *)(a3 + 8) )
    v19 = *(_QWORD *)(a3 + 8);
  v20 = 255;
  if ( v19 )
  {
    _BitScanReverse64(&v19, v19);
    v20 = 63 - (v19 ^ 0x3F);
  }
  *(_WORD *)(v14 + 2) = v20 | *(_WORD *)(v14 + 2) & 0xFFC0;
  v21 = *(_QWORD *)(a1 + 464);
  v48 = 257;
  if ( v21 != *(_QWORD *)(v14 + 8) )
  {
    if ( *(_BYTE *)v14 > 0x15u )
    {
      v50 = 257;
      v14 = sub_B52210(v14, v21, (__int64)v49, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
        a2[11],
        v14,
        v47,
        a2[7],
        a2[8]);
      v39 = 16LL * *((unsigned int *)a2 + 2);
      v40 = *a2;
      v41 = v40 + v39;
      while ( v41 != v40 )
      {
        v42 = *(_QWORD *)(v40 + 8);
        v43 = *(_DWORD *)v40;
        v40 += 16;
        sub_B99FD0(v14, v43, v42);
      }
    }
    else
    {
      v22 = (_BYTE *)(*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2[10] + 136LL))(
                       a2[10],
                       v14,
                       v21);
      v14 = (__int64)v22;
      if ( *v22 > 0x1Cu )
      {
        (*(void (__fastcall **)(__int64, _BYTE *, _QWORD *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
          a2[11],
          v22,
          v47,
          a2[7],
          a2[8]);
        v23 = *a2 + 16LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v23 )
        {
          v24 = *a2;
          do
          {
            v25 = *(_QWORD *)(v24 + 8);
            v26 = *(_DWORD *)v24;
            v24 += 16;
            sub_B99FD0(v14, v26, v25);
          }
          while ( v23 != v24 );
        }
      }
    }
  }
  return v14;
}
