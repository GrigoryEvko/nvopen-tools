// Function: sub_1A99820
// Address: 0x1a99820
//
void __fastcall sub_1A99820(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rbx
  _QWORD *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int8 v11; // cl
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  unsigned int v16; // esi
  __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned int v19; // ecx
  unsigned __int8 *v20; // rax
  __int64 v21; // r10
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int8 *v26; // rsi
  __int64 **v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // r14
  unsigned __int64 v30; // rdx
  __int64 *v31; // r14
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rsi
  __int64 v37; // rax
  int v38; // r11d
  unsigned __int8 *v39; // r9
  int v40; // eax
  int v41; // ecx
  __int64 *v42; // r15
  __int64 v43; // [rsp-F8h] [rbp-F8h] BYREF
  unsigned __int8 *v44; // [rsp-F0h] [rbp-F0h] BYREF
  unsigned __int8 **v45; // [rsp-E8h] [rbp-E8h] BYREF
  __int16 v46; // [rsp-D8h] [rbp-D8h]
  _WORD v47[16]; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned __int8 *v48[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v49; // [rsp-98h] [rbp-98h] BYREF
  unsigned __int8 *v50; // [rsp-88h] [rbp-88h] BYREF
  __int64 v51; // [rsp-80h] [rbp-80h]
  __int64 *v52; // [rsp-78h] [rbp-78h]
  __int64 v53; // [rsp-70h] [rbp-70h]
  __int64 v54; // [rsp-68h] [rbp-68h]
  int v55; // [rsp-60h] [rbp-60h]
  __int64 v56; // [rsp-58h] [rbp-58h]
  __int64 v57; // [rsp-50h] [rbp-50h]

  if ( a1 )
  {
    v5 = a1;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = sub_1648700(v5);
        v7 = (__int64)v6;
        if ( *((_BYTE *)v6 + 16) == 78 )
        {
          v8 = *(v6 - 3);
          if ( !*(_BYTE *)(v8 + 16) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 && *(_DWORD *)(v8 + 36) == 76 )
            break;
        }
LABEL_3:
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          return;
      }
      v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      v10 = *(_QWORD *)(v7 - 24 * v9);
      v11 = *(_BYTE *)(v10 + 16);
      if ( v11 == 88 )
      {
        v37 = sub_157F120(*(_QWORD *)(v10 + 40));
        v10 = sub_157EBA0(v37);
        v11 = *(_BYTE *)(v10 + 16);
        v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      }
      if ( v11 <= 0x17u )
        break;
      if ( v11 == 78 )
      {
        v30 = v10 | 4;
      }
      else
      {
        v12 = 0;
        if ( v11 != 29 )
          goto LABEL_13;
        v30 = v10 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v12 = v30 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = (v30 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
      if ( (v30 & 4) == 0 )
        goto LABEL_13;
LABEL_14:
      v14 = *(_QWORD *)(v7 + 24 * (2 - v9));
      v15 = *(_QWORD **)(v14 + 24);
      if ( *(_DWORD *)(v14 + 32) > 0x40u )
        v15 = (_QWORD *)*v15;
      v16 = *(_DWORD *)(a3 + 24);
      v17 = *(_QWORD *)(v13 + 24LL * (unsigned int)v15);
      v43 = v17;
      if ( !v16 )
      {
        ++*(_QWORD *)a3;
LABEL_58:
        v16 *= 2;
LABEL_59:
        sub_176F940(a3, v16);
        sub_176A9A0(a3, &v43, &v50);
        v39 = v50;
        v17 = v43;
        v41 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_54;
      }
      v18 = *(_QWORD *)(a3 + 8);
      v19 = (v16 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v20 = (unsigned __int8 *)(v18 + 16LL * v19);
      v21 = *(_QWORD *)v20;
      if ( v17 == *(_QWORD *)v20 )
      {
        v22 = *((_QWORD *)v20 + 1);
        goto LABEL_19;
      }
      v38 = 1;
      v39 = 0;
      while ( v21 != -8 )
      {
        if ( v39 || v21 != -16 )
          v20 = v39;
        v19 = (v16 - 1) & (v38 + v19);
        v42 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v42;
        if ( v17 == *v42 )
        {
          v22 = v42[1];
          goto LABEL_19;
        }
        ++v38;
        v39 = v20;
        v20 = (unsigned __int8 *)(v18 + 16LL * v19);
      }
      if ( !v39 )
        v39 = v20;
      v40 = *(_DWORD *)(a3 + 16);
      ++*(_QWORD *)a3;
      v41 = v40 + 1;
      if ( 4 * (v40 + 1) >= 3 * v16 )
        goto LABEL_58;
      if ( v16 - *(_DWORD *)(a3 + 20) - v41 <= v16 >> 3 )
        goto LABEL_59;
LABEL_54:
      *(_DWORD *)(a3 + 16) = v41;
      if ( *(_QWORD *)v39 != -8 )
        --*(_DWORD *)(a3 + 20);
      *(_QWORD *)v39 = v17;
      v22 = 0;
      *((_QWORD *)v39 + 1) = 0;
LABEL_19:
      v23 = *(_QWORD *)(v7 + 32);
      if ( v23 == *(_QWORD *)(v7 + 40) + 40LL || !v23 )
      {
        v3 = sub_16498A0(0);
        v50 = 0;
        v52 = 0;
        v53 = v3;
        v54 = 0;
        v55 = 0;
        v56 = 0;
        v57 = 0;
        v51 = 0;
        BUG();
      }
      v24 = sub_16498A0(v23 - 24);
      v50 = 0;
      v53 = v24;
      v54 = 0;
      v55 = 0;
      v56 = 0;
      v57 = 0;
      v25 = *(_QWORD *)(v23 + 16);
      v52 = (__int64 *)v23;
      v51 = v25;
      v26 = *(unsigned __int8 **)(v23 + 24);
      v48[0] = v26;
      if ( v26 )
      {
        sub_1623A60((__int64)v48, (__int64)v26, 2);
        if ( v50 )
          sub_161E7C0((__int64)&v50, (__int64)v50);
        v50 = v48[0];
        if ( v48[0] )
          sub_1623210((__int64)v48, v48[0], (__int64)&v50);
      }
      sub_1A956E0((__int64 *)v48, v7, (__int64)".casted", 7, byte_3F871B3, 0);
      v45 = v48;
      v27 = *(__int64 ***)(v22 + 56);
      v46 = 260;
      if ( v27 != *(__int64 ***)v7 )
      {
        if ( *(_BYTE *)(v7 + 16) > 0x10u )
        {
          v47[8] = 257;
          v7 = sub_15FDBD0(47, v7, (__int64)v27, (__int64)v47, 0);
          if ( v51 )
          {
            v31 = v52;
            sub_157E9D0(v51 + 40, v7);
            v32 = *(_QWORD *)(v7 + 24);
            v33 = *v31;
            *(_QWORD *)(v7 + 32) = v31;
            v33 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v7 + 24) = v33 | v32 & 7;
            *(_QWORD *)(v33 + 8) = v7 + 24;
            *v31 = *v31 & 7 | (v7 + 24);
          }
          sub_164B780(v7, (__int64 *)&v45);
          if ( v50 )
          {
            v44 = v50;
            sub_1623A60((__int64)&v44, (__int64)v50, 2);
            v34 = *(_QWORD *)(v7 + 48);
            v35 = v7 + 48;
            if ( v34 )
            {
              sub_161E7C0(v7 + 48, v34);
              v35 = v7 + 48;
            }
            v36 = v44;
            *(_QWORD *)(v7 + 48) = v44;
            if ( v36 )
              sub_1623210((__int64)&v44, v36, v35);
          }
        }
        else
        {
          v7 = sub_15A46C0(47, (__int64 ***)v7, v27, 0);
        }
      }
      if ( (__int64 *)v48[0] != &v49 )
        j_j___libc_free_0(v48[0], v49 + 1);
      v28 = sub_1648A60(64, 2u);
      v29 = (__int64)v28;
      if ( v28 )
        sub_15F9650((__int64)v28, v7, v22, 0, 0);
      sub_15F2180(v29, v7);
      if ( !v50 )
        goto LABEL_3;
      sub_161E7C0((__int64)&v50, (__int64)v50);
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        return;
    }
    v12 = 0;
LABEL_13:
    v13 = v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
    goto LABEL_14;
  }
}
