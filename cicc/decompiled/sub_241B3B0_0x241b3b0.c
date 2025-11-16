// Function: sub_241B3B0
// Address: 0x241b3b0
//
_QWORD *__fastcall sub_241B3B0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  _BYTE *v6; // rcx
  char v7; // dl
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *result; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r13
  unsigned __int8 v19; // al
  unsigned int v20; // edx
  _QWORD *v21; // rax
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v25; // rbx
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r9
  __int64 v31; // r8
  int v32; // r10d
  __int64 v33; // rdi
  __int64 *v34; // rdx
  unsigned int v35; // r11d
  __int64 *v36; // rax
  __int64 v37; // rcx
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // r10
  __int64 v43; // rcx
  unsigned int v44; // r9d
  __int64 *v45; // rax
  __int64 v46; // rdx
  __int64 *v47; // r11
  int v48; // ecx
  int v49; // edx
  __int64 *v50; // [rsp+8h] [rbp-148h]
  unsigned __int8 v51; // [rsp+10h] [rbp-140h]
  __int64 v53; // [rsp+10h] [rbp-140h]
  __int64 v54; // [rsp+10h] [rbp-140h]
  __int64 v55; // [rsp+18h] [rbp-138h]
  unsigned int v56; // [rsp+18h] [rbp-138h]
  __int64 v58; // [rsp+18h] [rbp-138h]
  int v59; // [rsp+18h] [rbp-138h]
  __int64 v60; // [rsp+28h] [rbp-128h] BYREF
  _QWORD v61[4]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v62; // [rsp+50h] [rbp-100h]
  _QWORD v63[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v64; // [rsp+80h] [rbp-D0h]
  __int64 v65; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v66; // [rsp+98h] [rbp-B8h]
  __int64 v67; // [rsp+C0h] [rbp-90h]
  __int64 v68; // [rsp+C8h] [rbp-88h]
  __int64 v69; // [rsp+D0h] [rbp-80h]
  __int64 v70; // [rsp+E8h] [rbp-68h]

  v3 = a2;
  v4 = a2;
  v5 = *(_QWORD *)(a2 + 16);
  if ( !v5 )
  {
LABEL_11:
    sub_23D0AB0((__int64)&v65, a2, 0, 0, 0);
    v17 = *a1;
    v62 = 257;
    v55 = *(_QWORD *)(*(_QWORD *)v17 + 48LL);
    v18 = sub_AA4E30(v67);
    v50 = (__int64 *)v55;
    v19 = sub_AE5260(v18, v55);
    v20 = *(_DWORD *)(v18 + 4);
    v51 = v19;
    v64 = 257;
    v56 = v20;
    v21 = sub_BD2C40(80, unk_3F10A14);
    v22 = (__int64)v21;
    if ( v21 )
      sub_B4CCA0((__int64)v21, v50, v56, 0, v51, (__int64)v63, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v70 + 16LL))(
      v70,
      v22,
      v61,
      v68,
      v69);
    v23 = 16LL * v66;
    if ( v65 != v65 + v23 )
    {
      v24 = v65 + v23;
      v25 = v65;
      do
      {
        v26 = *(_QWORD *)(v25 + 8);
        v27 = *(_DWORD *)v25;
        v25 += 16;
        sub_B99FD0(v22, v27, v26);
      }
      while ( v24 != v25 );
      v3 = a2;
      v4 = a2;
    }
    v28 = *a1;
    v60 = v4;
    v29 = *(unsigned int *)(v28 + 264);
    v58 = v28;
    v30 = v28 + 240;
    if ( (_DWORD)v29 )
    {
      v31 = (unsigned int)(v29 - 1);
      v32 = 1;
      v33 = *(_QWORD *)(v28 + 248);
      v34 = 0;
      v35 = v31 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v36 = (__int64 *)(v33 + 16LL * v35);
      v37 = *v36;
      if ( v4 == *v36 )
      {
LABEL_19:
        v38 = v36 + 1;
        goto LABEL_20;
      }
      while ( v37 != -4096 )
      {
        if ( !v34 && v37 == -8192 )
          v34 = v36;
        v35 = v31 & (v32 + v35);
        v36 = (__int64 *)(v33 + 16LL * v35);
        v37 = *v36;
        if ( v4 == *v36 )
          goto LABEL_19;
        ++v32;
      }
      if ( !v34 )
        v34 = v36;
      v63[0] = v34;
      ++*(_QWORD *)(v58 + 240);
      v37 = (unsigned int)(*(_DWORD *)(v58 + 256) + 1);
      if ( 4 * (int)v37 < (unsigned int)(3 * v29) )
      {
        v31 = (unsigned int)v29 >> 3;
        v33 = v4;
        if ( (int)v29 - *(_DWORD *)(v58 + 260) - (int)v37 > (unsigned int)v31 )
        {
LABEL_35:
          *(_DWORD *)(v58 + 256) = v37;
          if ( *v34 != -4096 )
            --*(_DWORD *)(v58 + 260);
          *v34 = v33;
          v38 = v34 + 1;
          v34[1] = 0;
LABEL_20:
          *v38 = v22;
          if ( !(unsigned __int8)sub_240D530(v33, v29, v34, v37, v31) )
          {
LABEL_21:
            sub_F94A20(&v65, v29);
            goto LABEL_5;
          }
          v63[0] = "_dfsa";
          v39 = *a1;
          v64 = 259;
          v40 = sub_23DEB90(&v65, *(__int64 **)(*(_QWORD *)v39 + 24LL), 0, (__int64)v63);
          v41 = *a1;
          v60 = v4;
          v42 = v40;
          v29 = *(unsigned int *)(v41 + 296);
          if ( (_DWORD)v29 )
          {
            v43 = *(_QWORD *)(v41 + 280);
            v44 = (v29 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v45 = (__int64 *)(v43 + 16LL * v44);
            v46 = *v45;
            if ( *v45 == v4 )
            {
LABEL_24:
              v45[1] = v42;
              goto LABEL_21;
            }
            v59 = 1;
            v47 = 0;
            while ( v46 != -4096 )
            {
              if ( !v47 && v46 == -8192 )
                v47 = v45;
              v44 = (v29 - 1) & (v59 + v44);
              v45 = (__int64 *)(v43 + 16LL * v44);
              v46 = *v45;
              if ( v4 == *v45 )
                goto LABEL_24;
              ++v59;
            }
            if ( v47 )
              v45 = v47;
            v61[0] = v45;
            v48 = *(_DWORD *)(v41 + 288);
            ++*(_QWORD *)(v41 + 272);
            v49 = v48 + 1;
            if ( 4 * (v48 + 1) < (unsigned int)(3 * v29) )
            {
              if ( (int)v29 - *(_DWORD *)(v41 + 292) - v49 > (unsigned int)v29 >> 3 )
              {
LABEL_47:
                *(_DWORD *)(v41 + 288) = v49;
                if ( *v45 != -4096 )
                  --*(_DWORD *)(v41 + 292);
                *v45 = v4;
                v45[1] = 0;
                goto LABEL_24;
              }
              v54 = v42;
LABEL_51:
              sub_241B1D0(v41 + 272, v29);
              v29 = (__int64)&v60;
              sub_2414530(v41 + 272, &v60, v61);
              v4 = v60;
              v42 = v54;
              v49 = *(_DWORD *)(v41 + 288) + 1;
              v45 = (__int64 *)v61[0];
              goto LABEL_47;
            }
          }
          else
          {
            v61[0] = 0;
            ++*(_QWORD *)(v41 + 272);
          }
          v54 = v42;
          LODWORD(v29) = 2 * v29;
          goto LABEL_51;
        }
LABEL_40:
        v53 = v30;
        sub_241B1D0(v30, v29);
        v29 = (__int64)&v60;
        sub_2414530(v53, &v60, v63);
        v33 = v60;
        v34 = (__int64 *)v63[0];
        v37 = (unsigned int)(*(_DWORD *)(v58 + 256) + 1);
        goto LABEL_35;
      }
    }
    else
    {
      v63[0] = 0;
      ++*(_QWORD *)(v28 + 240);
    }
    LODWORD(v29) = 2 * v29;
    goto LABEL_40;
  }
  while ( 1 )
  {
    v6 = *(_BYTE **)(v5 + 24);
    v7 = *v6;
    if ( *v6 <= 0x1Cu )
      break;
    if ( v7 != 61 )
    {
      if ( v7 != 62 )
        break;
      v16 = *((_QWORD *)v6 - 4);
      if ( !v16 || v16 != a2 )
        break;
    }
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      goto LABEL_11;
  }
LABEL_5:
  v8 = *a1 + 176;
  v9 = *(_QWORD *)(*(_QWORD *)*a1 + 72LL);
  v65 = v3;
  *sub_FAA780(v8, &v65) = v9;
  v10 = *a1;
  v11 = *(_QWORD *)(*(_QWORD *)*a1 + 40LL);
  result = (_QWORD *)sub_240D530(v8, &v65, v12, v13, v14);
  if ( (_BYTE)result )
  {
    v65 = v3;
    result = sub_FAA780(v10 + 208, &v65);
    *result = v11;
  }
  return result;
}
