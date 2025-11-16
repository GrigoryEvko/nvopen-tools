// Function: sub_11E40C0
// Address: 0x11e40c0
//
__int64 __fastcall sub_11E40C0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v5; // edx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int8 *v19; // r13
  __int64 v20; // rax
  int v21; // r10d
  __int64 v22; // rdi
  __int64 (__fastcall *v23)(__int64, __int64, unsigned __int8 *, unsigned __int8 *, _QWORD); // rax
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // r10d
  unsigned int *v29; // r13
  __int64 v30; // r14
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 result; // rax
  __int64 v35; // r15
  __int64 v36; // rax
  int v37; // [rsp+Ch] [rbp-F4h]
  unsigned __int8 *v38; // [rsp+10h] [rbp-F0h]
  __int64 v39; // [rsp+28h] [rbp-D8h]
  __int64 v40; // [rsp+30h] [rbp-D0h]
  _BYTE v41[32]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v42; // [rsp+60h] [rbp-A0h]
  __int64 v43[4]; // [rsp+70h] [rbp-90h] BYREF
  char v44; // [rsp+90h] [rbp-70h]
  char v45; // [rsp+91h] [rbp-6Fh]
  _QWORD v46[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v47; // [rsp+C0h] [rbp-40h]

  v5 = *a2;
  if ( v5 == 40 )
  {
    v6 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v6 = 0;
    if ( v5 != 85 )
    {
      v6 = 64;
      if ( v5 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v7 = sub_BD2BC0((__int64)a2);
  v9 = v7 + v8;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
LABEL_54:
      BUG();
LABEL_10:
    v13 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v9 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_54;
  v10 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v11 = sub_BD2BC0((__int64)a2);
  v13 = 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
LABEL_11:
  v14 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  if ( (unsigned int)((32 * v14 - 32 - v6 - v13) >> 5) == 1 )
  {
    if ( sub_B45190((__int64)a2) )
    {
      v47 = 259;
      v35 = *(_QWORD *)&a2[-32 * v14];
      v46[0] = "real";
      LODWORD(v43[0]) = 0;
      v15 = sub_94D3D0((unsigned int **)a3, v35, (__int64)v43, 1, (__int64)v46);
      v46[0] = "imag";
      v47 = 259;
      LODWORD(v43[0]) = 1;
      v16 = sub_94D3D0((unsigned int **)a3, v35, (__int64)v43, 1, (__int64)v46);
LABEL_18:
      v47 = 257;
      LODWORD(v43[0]) = sub_B45210((__int64)a2);
      BYTE4(v43[0]) = 1;
      v18 = sub_A826E0((unsigned int **)a3, (_BYTE *)v15, (_BYTE *)v15, v43[0], (__int64)v46, 0);
      v47 = 257;
      v19 = (unsigned __int8 *)v18;
      LODWORD(v43[0]) = sub_B45210((__int64)a2);
      BYTE4(v43[0]) = 1;
      v20 = sub_A826E0((unsigned int **)a3, (_BYTE *)v16, (_BYTE *)v16, v43[0], (__int64)v46, 0);
      v45 = 1;
      v38 = (unsigned __int8 *)v20;
      v43[0] = (__int64)"cabs";
      v44 = 3;
      BYTE4(v40) = 1;
      LODWORD(v40) = sub_B45210((__int64)a2);
      v42 = 257;
      BYTE4(v39) = 1;
      LODWORD(v39) = sub_B45210((__int64)a2);
      v21 = v39;
      if ( *(_BYTE *)(a3 + 108) )
      {
        v25 = sub_B35400(a3, 0x66u, (__int64)v19, (__int64)v38, v39, (__int64)v41, 0, 0, 0);
        goto LABEL_45;
      }
      v22 = *(_QWORD *)(a3 + 80);
      v23 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int8 *, unsigned __int8 *, _QWORD))(*(_QWORD *)v22 + 40LL);
      if ( (char *)v23 == (char *)sub_928A40 )
      {
        if ( *v19 > 0x15u || *v38 > 0x15u )
        {
LABEL_26:
          v37 = v21;
          v47 = 257;
          v26 = sub_B504D0(14, (__int64)v19, (__int64)v38, (__int64)v46, 0, 0);
          v27 = *(_QWORD *)(a3 + 96);
          v28 = v37;
          v25 = v26;
          if ( v27 )
          {
            sub_B99FD0(v26, 3u, v27);
            v28 = v37;
          }
          sub_B45150(v25, v28);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v25,
            v41,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v29 = *(unsigned int **)a3;
          v30 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v30 )
          {
            do
            {
              v31 = *((_QWORD *)v29 + 1);
              v32 = *v29;
              v29 += 4;
              sub_B99FD0(v25, v32, v31);
            }
            while ( (unsigned int *)v30 != v29 );
          }
LABEL_45:
          result = sub_B33BC0(a3, 0x14Fu, v25, v40, (__int64)v43);
          if ( !result || *(_BYTE *)result != 85 )
            return result;
LABEL_40:
          *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *((_WORD *)a2 + 1) & 3;
          return result;
        }
        if ( (unsigned __int8)sub_AC47B0(14) )
          v24 = sub_AD5570(14, (__int64)v19, v38, 0, 0);
        else
          v24 = sub_AABE40(0xEu, v19, v38);
        v21 = v39;
        v25 = v24;
      }
      else
      {
        v36 = v23(v22, 14, v19, v38, (unsigned int)v39);
        v21 = v39;
        v25 = v36;
      }
      if ( v25 )
        goto LABEL_45;
      goto LABEL_26;
    }
    return 0;
  }
  v15 = *(_QWORD *)&a2[-32 * v14];
  v16 = *(_QWORD *)&a2[32 * (1 - v14)];
  if ( *(_BYTE *)v15 == 18 )
  {
    if ( *(void **)(v15 + 24) == sub_C33340() )
      v17 = *(_QWORD *)(v15 + 32);
    else
      v17 = v15 + 24;
    if ( (*(_BYTE *)(v17 + 20) & 7) != 3 || !v16 )
      goto LABEL_17;
    v15 = v16;
    v46[0] = "cabs";
    v47 = 259;
  }
  else
  {
    if ( *(_BYTE *)v16 != 18
      || (*(void **)(v16 + 24) == sub_C33340() ? (v33 = *(_QWORD *)(v16 + 32)) : (v33 = v16 + 24),
          (*(_BYTE *)(v33 + 20) & 7) != 3) )
    {
LABEL_17:
      if ( sub_B45190((__int64)a2) )
        goto LABEL_18;
      return 0;
    }
    v46[0] = "cabs";
    v47 = 259;
  }
  LODWORD(v43[0]) = sub_B45210((__int64)a2);
  BYTE4(v43[0]) = 1;
  result = sub_B33BC0(a3, 0xAAu, v15, v43[0], (__int64)v46);
  if ( result && *(_BYTE *)result == 85 )
    goto LABEL_40;
  return result;
}
