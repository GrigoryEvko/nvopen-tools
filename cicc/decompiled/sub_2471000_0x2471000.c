// Function: sub_2471000
// Address: 0x2471000
//
__int64 __fastcall sub_2471000(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rbx
  unsigned __int64 v17; // r12
  __int64 v18; // r14
  unsigned __int8 v19; // al
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // r12
  char v27; // al
  __int64 v28; // rax
  _BYTE *v29; // r12
  __int64 v30; // rax
  _BYTE *v31; // rax
  unsigned __int64 v32; // r14
  int v33; // r11d
  __int64 v34; // rax
  _BYTE *v35; // r12
  _BYTE *v36; // rax
  __int64 v37; // rax
  unsigned int v38; // r11d
  unsigned __int64 v39; // r12
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 *v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // rax
  char v47; // r12
  char v48; // al
  __int64 v49; // r8
  unsigned int v50; // r11d
  __int64 v51; // rdx
  int v52; // eax
  int v53; // eax
  _BYTE *v54; // r12
  _BYTE *v55; // rax
  __int64 v56; // rax
  unsigned int v57; // r11d
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rax
  __int64 v61; // [rsp+8h] [rbp-C8h]
  unsigned int v62; // [rsp+10h] [rbp-C0h]
  unsigned int v63; // [rsp+10h] [rbp-C0h]
  unsigned int v64; // [rsp+14h] [rbp-BCh]
  __int64 v65; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v66; // [rsp+20h] [rbp-B0h]
  __int64 *v67; // [rsp+20h] [rbp-B0h]
  __int64 v68; // [rsp+20h] [rbp-B0h]
  unsigned int v69; // [rsp+28h] [rbp-A8h]
  unsigned int v70; // [rsp+2Ch] [rbp-A4h]
  char v71; // [rsp+38h] [rbp-98h]
  int v72; // [rsp+38h] [rbp-98h]
  __int64 *v73; // [rsp+40h] [rbp-90h]
  unsigned int v74; // [rsp+50h] [rbp-80h]
  unsigned __int8 v75; // [rsp+56h] [rbp-7Ah]
  unsigned __int64 v76; // [rsp+58h] [rbp-78h]
  int v77; // [rsp+68h] [rbp-68h]
  const char *v78; // [rsp+70h] [rbp-60h] BYREF
  __int64 v79; // [rsp+78h] [rbp-58h]
  __int16 v80; // [rsp+90h] [rbp-40h]

  v4 = a1;
  v5 = sub_B2BEC0(*(_QWORD *)(a1 + 8));
  v6 = *a2;
  v65 = v5;
  if ( v6 == 40 )
  {
    v7 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v7 = -32;
    if ( v6 != 85 )
    {
      v7 = -96;
      if ( v6 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_9;
  v8 = sub_BD2BC0((__int64)a2);
  v10 = v8 + v9;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v10 >> 4) )
      goto LABEL_64;
  }
  else if ( (unsigned int)((v10 - sub_BD2BC0((__int64)a2)) >> 4) )
  {
    if ( (a2[7] & 0x80u) != 0 )
    {
      v11 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v12 = sub_BD2BC0((__int64)a2);
      v7 -= 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
      goto LABEL_9;
    }
LABEL_64:
    BUG();
  }
LABEL_9:
  v73 = (__int64 *)&a2[v7];
  v14 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( &a2[-v14] == &a2[v7] )
  {
    v20 = 0;
    goto LABEL_33;
  }
  v15 = (__int64 *)&a2[-v14];
  v76 = 0;
  v64 = 160;
  v69 = 0;
  v70 = 128;
  v74 = 16;
  do
  {
    v17 = (unsigned int)(*(_DWORD *)(*((_QWORD *)a2 + 10) + 12LL) - 1);
    v18 = *(_QWORD *)(*v15 + 8);
    if ( sub_BCAC40(v18, 128) || (v19 = *(_BYTE *)(v18 + 8), v19 == 5) )
    {
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 96LL);
LABEL_25:
      if ( v74 <= 0x37 )
      {
        if ( v17 > v76 )
        {
          v74 += 8;
          goto LABEL_14;
        }
        v47 = sub_B49B80((__int64)a2, v76, 79);
        v48 = sub_B49B80((__int64)a2, v76, 54);
        v49 = v74;
        v50 = v74 + 8;
        if ( v47 )
        {
          v72 = 1;
        }
        else if ( v48 )
        {
          v72 = 2;
        }
        else
        {
          v78 = (const char *)sub_BDB740(v65, v18);
          v79 = v51;
          v52 = sub_CA1930(&v78);
          v50 = v74 + 8;
          v72 = 0;
          v53 = v74 - v52;
          v49 = (unsigned int)(v53 + 8);
          v74 = v53 + 8;
        }
        v63 = v50;
        v80 = 257;
        v68 = v49;
        v54 = sub_94BCF0(
                (unsigned int **)a3,
                *(_QWORD *)(*(_QWORD *)(a1 + 16) + 136LL),
                *(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL),
                (__int64)&v78);
        v80 = 257;
        v55 = (_BYTE *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL), v68, 0);
        v56 = sub_929C50((unsigned int **)a3, v54, v55, (__int64)&v78, 0, 0);
        v57 = v63;
        v61 = 0;
        v32 = v56;
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
        {
          v59 = sub_24648B0(a1, (unsigned int **)a3, v74);
          v57 = v63;
          v61 = v59;
        }
        v74 = v57;
        goto LABEL_48;
      }
      goto LABEL_13;
    }
    if ( v19 <= 3u || (v19 & 0xFD) == 4 )
    {
      if ( *(_BYTE *)(a1 + 180) )
        goto LABEL_25;
      if ( v70 <= 0x9F )
      {
        if ( v17 > v76 )
        {
          v70 += 8;
          goto LABEL_14;
        }
        v34 = *(_QWORD *)(a1 + 16);
        v80 = 257;
        v35 = sub_94BCF0((unsigned int **)a3, *(_QWORD *)(v34 + 136), *(_QWORD *)(v34 + 80), (__int64)&v78);
        v80 = 257;
        v36 = (_BYTE *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL), v70, 0);
        v37 = sub_929C50((unsigned int **)a3, v35, v36, (__int64)&v78, 0, 0);
        v38 = v70 + 8;
        v61 = 0;
        v32 = v37;
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
        {
          v58 = sub_24648B0(a1, (unsigned int **)a3, v70);
          v38 = v70 + 8;
          v61 = v58;
        }
        v70 = v38;
        v72 = 0;
        goto LABEL_48;
      }
    }
    else
    {
      if ( (v19 & 0xFD) == 0xC )
        goto LABEL_25;
      if ( (unsigned __int8)(v19 - 17) <= 1u && v69 <= 7 )
      {
        if ( v17 > v76 )
        {
          ++v69;
          goto LABEL_14;
        }
        goto LABEL_34;
      }
    }
LABEL_13:
    if ( v17 > v76 )
      goto LABEL_14;
LABEL_34:
    v78 = (const char *)sub_BDB740(v65, v18);
    v79 = v24;
    v25 = v64;
    v26 = sub_CA1930(&v78);
    if ( v64 + 8 * ((v26 != 0) + ((v26 - (unsigned __int64)(v26 != 0)) >> 3)) > 0x320 )
    {
      v64 = 800;
      goto LABEL_14;
    }
    v66 = 8 * ((v26 != 0) + ((v26 - (unsigned __int64)(v26 != 0)) >> 3));
    v71 = sub_B49B80((__int64)a2, v76, 79);
    v27 = sub_B49B80((__int64)a2, v76, 54);
    if ( v71 )
    {
      v72 = 1;
      v62 = v64;
    }
    else if ( v27 )
    {
      v72 = 2;
      v62 = v64;
    }
    else
    {
      v72 = 0;
      v62 = v64 + v66 - v26;
      v25 = v62;
    }
    v28 = *(_QWORD *)(a1 + 16);
    v80 = 257;
    v29 = sub_94BCF0((unsigned int **)a3, *(_QWORD *)(v28 + 136), *(_QWORD *)(v28 + 80), (__int64)&v78);
    v30 = *(_QWORD *)(a1 + 16);
    v80 = 257;
    v31 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v30 + 80), v25, 0);
    v61 = 0;
    v32 = sub_929C50((unsigned int **)a3, v29, v31, (__int64)&v78, 0, 0);
    v33 = v66;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
    {
      v60 = sub_24648B0(a1, (unsigned int **)a3, v62);
      v33 = v66;
      v61 = v60;
    }
    v64 += v33;
LABEL_48:
    if ( v32 )
    {
      v39 = sub_246F3F0(*(_QWORD *)(a1 + 24), *v15);
      if ( v72 )
      {
        v67 = *(__int64 **)(a1 + 24);
        v46 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
        v39 = sub_2464970(v67, (unsigned int **)a3, v39, v46, v72 == 2);
      }
      v78 = "_msarg_va_s";
      v40 = *(_QWORD *)(a1 + 16);
      v80 = 259;
      v41 = sub_24633A0((__int64 *)a3, 0x30u, v32, *(__int64 ***)(v40 + 96), (__int64)&v78, 0, v77, 0);
      sub_2463EC0((__int64 *)a3, v39, v41, v75, 0);
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
      {
        v42 = sub_246EE10(*(_QWORD *)(a1 + 24), *v15);
        v43 = sub_9208B0(v65, *(_QWORD *)(v39 + 8));
        v44 = *(__int64 **)(a1 + 24);
        v79 = v45;
        v78 = (const char *)((unsigned __int64)(v43 + 7) >> 3);
        sub_24677C0(v44, a3, v42, v61, (__int64)v78, v45, byte_4FE8EA9);
      }
    }
LABEL_14:
    ++v76;
    v15 += 4;
  }
  while ( v73 != v15 );
  v4 = a1;
  v20 = v64 - 160;
LABEL_33:
  v21 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
  v22 = sub_ACD640(v21, v20, 0);
  return sub_2463EC0((__int64 *)a3, v22, *(_QWORD *)(*(_QWORD *)(v4 + 16) + 152LL), 0, 0);
}
