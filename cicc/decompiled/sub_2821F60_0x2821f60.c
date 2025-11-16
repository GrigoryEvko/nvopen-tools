// Function: sub_2821F60
// Address: 0x2821f60
//
__int64 __fastcall sub_2821F60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r14
  __int64 v7; // rdi
  unsigned int v8; // r15d
  int v9; // eax
  unsigned int v10; // r9d
  unsigned __int8 *v11; // r14
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 *v15; // r11
  __int64 v16; // rax
  __int64 v17; // rax
  bool v18; // al
  bool v19; // al
  __int64 v20; // rcx
  __int64 v21; // r8
  char v22; // r15
  __int64 v23; // r11
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  char v28; // al
  __int64 *v29; // rax
  unsigned int v30; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  bool v34; // al
  __int64 v35; // r11
  unsigned int v36; // edx
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rsi
  bool v40; // zf
  unsigned __int64 v41; // rdx
  unsigned int v42; // edx
  bool v43; // al
  bool v44; // al
  __int64 *v45; // rax
  unsigned __int64 v46; // rcx
  unsigned int v47; // eax
  unsigned __int64 v48; // [rsp+0h] [rbp-B0h]
  __int64 v49; // [rsp+8h] [rbp-A8h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 *v51; // [rsp+10h] [rbp-A0h]
  __int64 v52; // [rsp+10h] [rbp-A0h]
  __int64 v53; // [rsp+10h] [rbp-A0h]
  bool v54; // [rsp+10h] [rbp-A0h]
  __int64 *v55; // [rsp+18h] [rbp-98h]
  __int64 v56; // [rsp+18h] [rbp-98h]
  __int64 v57; // [rsp+18h] [rbp-98h]
  unsigned int v58; // [rsp+18h] [rbp-98h]
  unsigned int v59; // [rsp+18h] [rbp-98h]
  __int64 *v60; // [rsp+18h] [rbp-98h]
  __int64 v61; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v62; // [rsp+20h] [rbp-90h]
  unsigned int v63; // [rsp+20h] [rbp-90h]
  __int64 v64; // [rsp+28h] [rbp-88h]
  __int64 v65; // [rsp+28h] [rbp-88h]
  unsigned __int16 v66; // [rsp+28h] [rbp-88h]
  unsigned __int8 v67; // [rsp+28h] [rbp-88h]
  _QWORD *v68; // [rsp+28h] [rbp-88h]
  __int64 v69; // [rsp+28h] [rbp-88h]
  unsigned __int64 v70; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v71; // [rsp+38h] [rbp-78h]
  unsigned __int64 v72; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v73; // [rsp+48h] [rbp-68h]
  unsigned __int64 v74; // [rsp+50h] [rbp-60h] BYREF
  __int64 *v75; // [rsp+58h] [rbp-58h]
  __int64 v76; // [rsp+60h] [rbp-50h]
  int v77; // [rsp+68h] [rbp-48h]
  char v78; // [rsp+6Ch] [rbp-44h]
  __int64 v79; // [rsp+70h] [rbp-40h] BYREF

  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v7 = *(_QWORD *)(a2 + 32 * (3 - v6));
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 <= 0x40 )
  {
    if ( *(_QWORD *)(v7 + 24) )
      return 0;
  }
  else
  {
    v9 = sub_C444A0(v7 + 24);
    v10 = 0;
    if ( v8 != v9 )
      return v10;
  }
  v10 = *(unsigned __int8 *)(a1 + 264);
  if ( !(_BYTE)v10 )
    return v10;
  if ( unk_4FFF908 )
    return 0;
  v11 = sub_BD3990(*(unsigned __int8 **)(a2 - 32 * v6), a2);
  v12 = sub_DD8400(*(_QWORD *)(a1 + 32), (__int64)v11);
  v10 = 0;
  if ( *((_WORD *)v12 + 12) != 8 )
    return v10;
  if ( v12[6] != *(_QWORD *)a1 )
    return v10;
  if ( v12[5] != 2 )
    return v10;
  v55 = v12;
  v64 = *(_QWORD *)(v12[4] + 8);
  v13 = sub_DD8400(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v10 = 0;
  v14 = (__int64)v13;
  if ( v13 == 0 || v64 == 0 )
    return v10;
  v15 = v55;
  v16 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v16 == 17 )
  {
    if ( *(_DWORD *)(v16 + 32) <= 0x40u )
      v68 = *(_QWORD **)(v16 + 24);
    else
      v68 = **(_QWORD ***)(v16 + 24);
    v32 = *(_QWORD *)(v55[4] + 8);
    if ( !*(_WORD *)(v32 + 24) )
    {
      v33 = *(_QWORD *)(v32 + 32);
      v71 = *(_DWORD *)(v33 + 32);
      v58 = v71;
      if ( v71 > 0x40 )
      {
        v60 = v15;
        sub_C43780((__int64)&v70, (const void **)(v33 + 24));
        v63 = v71;
        v44 = sub_D94970((__int64)&v70, v68);
        v36 = v63;
        v35 = (__int64)v60;
        if ( v44 )
          goto LABEL_30;
        LODWORD(v75) = v63;
        if ( v63 > 0x40 )
        {
          sub_C43780((__int64)&v74, (const void **)&v70);
          v36 = (unsigned int)v75;
          v35 = (__int64)v60;
          if ( (unsigned int)v75 > 0x40 )
          {
            sub_C43D10((__int64)&v74);
            v35 = (__int64)v60;
LABEL_53:
            v49 = v35;
            sub_C46250((__int64)&v74);
            v47 = (unsigned int)v75;
            LODWORD(v75) = 0;
            v73 = v47;
            v72 = v74;
            v54 = sub_D94970((__int64)&v72, v68);
            sub_969240((__int64 *)&v72);
            sub_969240((__int64 *)&v74);
            v35 = v49;
            if ( !v54 )
            {
              if ( v71 > 0x40 && v70 )
                j_j___libc_free_0_0(v70);
              return 0;
            }
            v36 = v71;
LABEL_30:
            LODWORD(v75) = v36;
            if ( v36 > 0x40 )
            {
              v53 = v35;
              sub_C43780((__int64)&v74, (const void **)&v70);
              v36 = (unsigned int)v75;
              v35 = v53;
              if ( (unsigned int)v75 > 0x40 )
              {
                sub_C43D10((__int64)&v74);
                v35 = v53;
LABEL_35:
                v52 = v35;
                sub_C46250((__int64)&v74);
                v42 = (unsigned int)v75;
                LODWORD(v75) = 0;
                v73 = v42;
                v59 = v42;
                v72 = v74;
                v48 = v74;
                v43 = sub_D94970((__int64)&v72, v68);
                v23 = v52;
                v10 = 0;
                v22 = v43;
                if ( v59 > 0x40 )
                {
                  if ( v48 )
                  {
                    j_j___libc_free_0_0(v48);
                    v23 = v52;
                    v10 = 0;
                    if ( (unsigned int)v75 > 0x40 )
                    {
                      if ( v74 )
                      {
                        j_j___libc_free_0_0(v74);
                        v10 = 0;
                        v23 = v52;
                      }
                    }
                  }
                }
                if ( v71 > 0x40 && v70 )
                {
                  v69 = v23;
                  j_j___libc_free_0_0(v70);
                  v23 = v69;
                  v10 = 0;
                }
                goto LABEL_18;
              }
              v37 = v74;
            }
            else
            {
              v37 = v70;
            }
            v38 = ~v37;
            v39 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v36;
            v40 = v36 == 0;
            v41 = 0;
            if ( !v40 )
              v41 = v39;
            v74 = v41 & v38;
            goto LABEL_35;
          }
LABEL_50:
          v46 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v36;
          if ( !v36 )
            v46 = 0;
          v74 = v46 & ~v74;
          goto LABEL_53;
        }
      }
      else
      {
        v51 = v15;
        v70 = *(_QWORD *)(v33 + 24);
        v34 = sub_D94970((__int64)&v70, v68);
        v35 = (__int64)v51;
        v36 = v58;
        if ( v34 )
          goto LABEL_30;
        LODWORD(v75) = v58;
      }
      v74 = v70;
      goto LABEL_50;
    }
    return 0;
  }
  v17 = *((_QWORD *)v11 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
    v17 = **(_QWORD **)(v17 + 16);
  if ( !(*(_DWORD *)(v17 + 8) >> 8) )
  {
    v50 = v14;
    v18 = sub_DADE90(*(_QWORD *)(a1 + 32), v14, *(_QWORD *)a1);
    v10 = 0;
    if ( v18 )
    {
      v19 = sub_D969D0(v64);
      v10 = 0;
      v21 = v50;
      v22 = v19;
      v23 = (__int64)v55;
      if ( v19 )
      {
        v45 = sub_DCAF50(*(__int64 **)(a1 + 32), v64, 0);
        v10 = 0;
        v21 = v50;
        v64 = (__int64)v45;
        v23 = (__int64)v55;
      }
      v24 = v64;
      v65 = v21;
      if ( v21 == v24
        || (v56 = v23,
            v61 = sub_DE4F70(*(__int64 **)(a1 + 32), v24, *(_QWORD *)a1),
            v25 = sub_DE4F70(*(__int64 **)(a1 + 32), v65, *(_QWORD *)a1),
            v23 = v56,
            v10 = 0,
            v61 == v25) )
      {
LABEL_18:
        v57 = v23;
        v26 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
        v27 = 32 * (1 - v26);
        if ( *(_QWORD *)(a2 + v27) )
        {
          v62 = *(unsigned __int8 **)(a2 + v27);
          v28 = sub_D48480(*(_QWORD *)a1, (__int64)v62, v26, v20);
          v10 = 0;
          if ( v28 )
          {
            v78 = 1;
            v75 = &v79;
            v76 = 0x100000001LL;
            v77 = 0;
            v79 = a2;
            v74 = 1;
            v66 = sub_A74840((_QWORD *)(a2 + 72), 0);
            v29 = sub_DD8400(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
            v30 = sub_2820F60(a1, (__int64)v11, (__int64)v29, v66, v62, a2, (__int64)&v74, v57, a3, v22, 1);
            v10 = v30;
            if ( !v78 )
            {
              v67 = v30;
              _libc_free((unsigned __int64)v75);
              return v67;
            }
          }
        }
      }
    }
  }
  return v10;
}
