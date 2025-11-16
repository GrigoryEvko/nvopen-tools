// Function: sub_28B3870
// Address: 0x28b3870
//
__int64 __fastcall sub_28B3870(_QWORD *a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned int v6; // r13d
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 v11; // rdx
  int v12; // ecx
  __int64 *v13; // rax
  _DWORD *v14; // rax
  __int64 v15; // r15
  unsigned __int8 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  __int64 v21; // r15
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r11
  int v28; // ecx
  __int64 v29; // rsi
  int v30; // ecx
  unsigned int v31; // edx
  unsigned __int8 **v32; // rax
  unsigned __int8 *v33; // rdi
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  int v39; // eax
  int v40; // r8d
  __int64 v41; // [rsp+0h] [rbp-100h]
  char v42; // [rsp+8h] [rbp-F8h]
  unsigned int v43; // [rsp+8h] [rbp-F8h]
  __int64 v44; // [rsp+10h] [rbp-F0h]
  __int64 v45; // [rsp+18h] [rbp-E8h]
  __int64 v46; // [rsp+18h] [rbp-E8h]
  int v47; // [rsp+2Ch] [rbp-D4h] BYREF
  unsigned __int64 v48; // [rsp+30h] [rbp-D0h] BYREF
  char v49; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v50[2]; // [rsp+40h] [rbp-C0h] BYREF
  char v51; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD *v52; // [rsp+88h] [rbp-78h]
  void *v53; // [rsp+C0h] [rbp-40h]

  if ( sub_B46500(a2) )
    return 0;
  v6 = a2[2] & 1;
  if ( (a2[2] & 1) != 0 || (a2[7] & 0x20) != 0 && sub_B91C10((__int64)a2, 9) )
    return 0;
  v8 = sub_B43CC0((__int64)a2);
  v9 = *((_QWORD *)a2 - 8);
  v10 = v8;
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(unsigned __int8 *)(v11 + 8);
  if ( (unsigned int)(v12 - 17) <= 1 )
  {
    v13 = *(__int64 **)(v11 + 16);
    v11 = *v13;
    LOBYTE(v12) = *(_BYTE *)(*v13 + 8);
  }
  if ( (_BYTE)v12 == 14 )
  {
    v45 = v10;
    v14 = sub_AE2980(v10, *(_DWORD *)(v11 + 8) >> 8);
    v10 = v45;
    if ( *((_BYTE *)v14 + 16) )
      return v6;
  }
  if ( *(_BYTE *)v9 != 61 )
  {
    if ( (*(_BYTE *)(*a1 + 53LL) & 4) == 0 && (*(_BYTE *)(*(_QWORD *)*a1 + 90LL) & 0x30) != 0 || (_BYTE)qword_5004468 )
    {
      v15 = *((_QWORD *)a2 - 8);
      v46 = v10;
      v16 = (unsigned __int8 *)sub_98A180((unsigned __int8 *)v15, v10);
      if ( v16 )
      {
        v44 = (__int64)v16;
        v17 = sub_28AD0D0((__int64)a1, (__int64)a2, *((_QWORD *)a2 - 4), v16);
        if ( v17 )
        {
          v6 = 1;
          *(_QWORD *)a3 = v17 + 24;
          *(_WORD *)(a3 + 8) = 0;
        }
        else
        {
          v18 = *(_QWORD *)(v15 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 15 <= 1 )
          {
            v50[0] = sub_9208B0(v46, v18);
            v50[1] = v19;
            v48 = (v50[0] + 7) >> 3;
            v49 = v19;
            if ( !(_BYTE)v19 )
            {
              sub_23D0AB0((__int64)v50, (__int64)a2, 0, 0, 0);
              _BitScanReverse64(&v20, 1LL << (*((_WORD *)a2 + 1) >> 1));
              v42 = v20 ^ 0x3F;
              v41 = sub_CA1930(&v48);
              v21 = *((_QWORD *)a2 - 4);
              v22 = (unsigned __int8)(63 - v42);
              BYTE1(v22) = 1;
              v43 = v22;
              v23 = sub_BCB2E0(v52);
              v24 = sub_ACD640(v23, v41, 0);
              v25 = sub_B34240((__int64)v50, v21, v44, v24, v43, 0, 0, 0, 0);
              v47 = 38;
              sub_B47C00(v25, (__int64)a2, &v47, 1);
              v26 = a1[5];
              v27 = 0;
              v28 = *(_DWORD *)(v26 + 56);
              v29 = *(_QWORD *)(v26 + 40);
              if ( v28 )
              {
                v30 = v28 - 1;
                v31 = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
                v32 = (unsigned __int8 **)(v29 + 16LL * v31);
                v33 = *v32;
                if ( a2 == *v32 )
                {
LABEL_21:
                  v27 = (__int64)v32[1];
                }
                else
                {
                  v39 = 1;
                  while ( v33 != (unsigned __int8 *)-4096LL )
                  {
                    v40 = v39 + 1;
                    v31 = v30 & (v39 + v31);
                    v32 = (unsigned __int8 **)(v29 + 16LL * v31);
                    v33 = *v32;
                    if ( a2 == *v32 )
                      goto LABEL_21;
                    v39 = v40;
                  }
                }
              }
              v34 = (__int64 *)sub_D69520((_QWORD *)a1[6], v25, 0, v27);
              sub_D75120((__int64 *)a1[6], v34, 0);
              sub_28AAD10((__int64)a1, a2, v35, v36, v37, v38);
              *(_QWORD *)a3 = v25 + 24;
              *(_WORD *)(a3 + 8) = 0;
              nullsub_61();
              v53 = &unk_49DA100;
              nullsub_63();
              if ( (char *)v50[0] != &v51 )
                _libc_free(v50[0]);
              return 1;
            }
          }
        }
      }
    }
    return v6;
  }
  return sub_28B2C50(a1, (__int64)a2, v9, v10, a3);
}
