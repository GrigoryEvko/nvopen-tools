// Function: sub_AED7C0
// Address: 0xaed7c0
//
unsigned __int64 __fastcall sub_AED7C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  unsigned __int64 v4; // r12
  __int64 v5; // rbx
  int v6; // ecx
  unsigned __int64 result; // rax
  __int64 v8; // rdx
  unsigned __int8 v9; // al
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  int v13; // r8d
  __int64 *v14; // rdi
  unsigned int v15; // esi
  __int64 *v16; // rax
  __int64 v17; // rdx
  unsigned __int8 v18; // al
  bool v19; // dl
  __int64 *v20; // rdi
  __int64 v21; // r13
  __int64 *v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rbx
  int v34; // r9d
  int v35; // ecx
  int v36; // eax
  __int64 v37; // rax
  int v38; // eax
  int v39; // eax
  _BYTE *v40; // rax
  _BYTE *v41; // rax
  __int64 v42; // rdx
  int v43; // esi
  __int64 *v44; // rax
  int v45; // r12d
  int v46; // eax
  __int64 v47; // rsi
  int v48; // eax
  int v49; // r13d
  int v50; // r8d
  int v51; // esi
  __int64 v52; // rax
  _QWORD *v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rax
  unsigned int v56; // eax
  unsigned int v57; // r13d
  __int64 *v58; // rax
  __int64 v59; // rdx
  __int64 *v60; // rbx
  __int64 *v61; // r12
  __int64 v62; // r11
  __int64 v63; // r13
  __int64 *v64; // rsi
  _QWORD *v65; // rdi
  int v66; // ecx
  int v67; // r10d
  __int64 v68; // [rsp+0h] [rbp-110h]
  __int64 v69; // [rsp+8h] [rbp-108h]
  __int64 v70; // [rsp+10h] [rbp-100h]
  int v71; // [rsp+1Ch] [rbp-F4h]
  __int64 v72; // [rsp+20h] [rbp-F0h]
  int v73; // [rsp+28h] [rbp-E8h]
  int v74; // [rsp+2Ch] [rbp-E4h]
  int v75; // [rsp+30h] [rbp-E0h]
  int v76; // [rsp+34h] [rbp-DCh]
  __int64 v77; // [rsp+38h] [rbp-D8h]
  int v78; // [rsp+40h] [rbp-D0h]
  int v79; // [rsp+44h] [rbp-CCh]
  __int64 v80; // [rsp+48h] [rbp-C8h]
  __int64 v81; // [rsp+50h] [rbp-C0h]
  __int64 v82; // [rsp+50h] [rbp-C0h]
  __int64 v83; // [rsp+58h] [rbp-B8h]
  __int64 v84; // [rsp+58h] [rbp-B8h]
  __int64 v85; // [rsp+60h] [rbp-B0h]
  __int64 v86; // [rsp+68h] [rbp-A8h]
  __int64 v87; // [rsp+70h] [rbp-A0h]
  __int64 v88; // [rsp+78h] [rbp-98h]
  int v89; // [rsp+78h] [rbp-98h]
  __int64 v90; // [rsp+88h] [rbp-88h] BYREF
  __int64 *v91; // [rsp+90h] [rbp-80h] BYREF
  __int64 v92; // [rsp+98h] [rbp-78h]
  _BYTE v93[112]; // [rsp+A0h] [rbp-70h] BYREF

  v3 = a2;
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v4 )
  {
    if ( !a2 )
    {
      v90 = 0;
      v12 = 0;
      goto LABEL_79;
    }
LABEL_5:
    v9 = *(_BYTE *)v3;
    if ( *(_BYTE *)v3 == 18 )
    {
      v10 = sub_A17150((_BYTE *)(v3 - 16));
      sub_AED7C0(a1, *((_QWORD *)v10 + 5));
      v11 = sub_AECE70(a1, v3);
      v5 = *(_QWORD *)(a1 + 8);
      LODWORD(v4) = *(_DWORD *)(a1 + 24);
      v12 = v11;
      goto LABEL_7;
    }
    if ( v9 == 15 )
    {
      v12 = *(_QWORD *)(a1 + 32);
      goto LABEL_7;
    }
    if ( v9 != 17 )
    {
      v12 = v3;
      if ( v9 == 16 )
        goto LABEL_7;
      if ( (unsigned int)v9 - 19 > 1 )
      {
        if ( v9 == 6 )
        {
          v44 = (__int64 *)sub_A17150((_BYTE *)(v3 - 16));
          v45 = sub_AE5C80(a1, *v44);
          if ( (*(_BYTE *)(v3 - 16) & 2) != 0 )
            v46 = *(_DWORD *)(v3 - 24);
          else
            v46 = (*(_WORD *)(v3 - 16) >> 6) & 0xF;
          v47 = 0;
          if ( v46 == 2 )
            v47 = *((_QWORD *)sub_A17150((_BYTE *)(v3 - 16)) + 1);
          v48 = sub_AE5C80(a1, v47);
          v49 = *(unsigned __int16 *)(v3 + 2);
          v50 = v48;
          v51 = *(_DWORD *)(v3 + 4);
          v52 = *(_QWORD *)(v3 + 8);
          v53 = (_QWORD *)(v52 & 0xFFFFFFFFFFFFFFF8LL);
          v54 = (v52 >> 2) & 1;
          if ( (*(_BYTE *)(v3 + 1) & 0x7F) == 1 )
          {
            if ( (_BYTE)v54 )
              v53 = (_QWORD *)*v53;
            v55 = sub_B01860((_DWORD)v53, v51, v49, v45, v50, 0, 1, 1);
          }
          else
          {
            if ( (_BYTE)v54 )
              v53 = (_QWORD *)*v53;
            v55 = sub_B01860((_DWORD)v53, v51, v49, v45, v50, 0, 0, 1);
          }
          v5 = *(_QWORD *)(a1 + 8);
          LODWORD(v4) = *(_DWORD *)(a1 + 24);
          v12 = v55;
        }
        else
        {
          if ( v9 > 0x1Eu )
          {
            v12 = 0;
            if ( (unsigned __int8)(v9 - 33) <= 3u )
              goto LABEL_7;
          }
          else
          {
            v12 = 0;
            if ( v9 > 8u )
              goto LABEL_7;
          }
          v91 = (__int64 *)v93;
          v92 = 0x800000000LL;
          if ( (*(_BYTE *)(v3 - 16) & 2) != 0 )
            v56 = *(_DWORD *)(v3 - 24);
          else
            v56 = (*(_WORD *)(v3 - 16) >> 6) & 0xF;
          v57 = 0;
          if ( v56 > 8 )
          {
            sub_C8D5F0(&v91, v93, v56, 8);
            v57 = v92;
          }
          v58 = (__int64 *)sub_A17150((_BYTE *)(v3 - 16));
          v60 = &v58[v59];
          if ( v58 != v60 )
          {
            v61 = v58;
            do
            {
              if ( *v61 )
              {
                v63 = sub_AE5C80(a1, *v61);
                if ( v62 + 1 > (unsigned __int64)HIDWORD(v92) )
                {
                  sub_C8D5F0(&v91, v93, v62 + 1, 8);
                  v62 = (unsigned int)v92;
                }
                v91[v62] = v63;
                v57 = v92 + 1;
                LODWORD(v92) = v92 + 1;
              }
              ++v61;
            }
            while ( v60 != v61 );
          }
          v64 = v91;
          v65 = (_QWORD *)(*(_QWORD *)(v3 + 8) & 0xFFFFFFFFFFFFFFF8LL);
          if ( (*(_QWORD *)(v3 + 8) & 4) != 0 )
            v65 = (_QWORD *)*v65;
          v12 = sub_B9C770(v65, v91, v57, 0, 1);
          if ( v91 != (__int64 *)v93 )
            _libc_free(v91, v64);
          v5 = *(_QWORD *)(a1 + 8);
          LODWORD(v4) = *(_DWORD *)(a1 + 24);
        }
      }
      else
      {
        v40 = sub_A17150((_BYTE *)(v3 - 16));
        v41 = (_BYTE *)sub_AE5C80(a1, *((_QWORD *)v40 + 1));
        v12 = (__int64)v41;
        if ( v41 && (unsigned __int8)(*v41 - 5) >= 0x20u )
          v12 = 0;
      }
LABEL_7:
      v90 = v3;
      if ( (_DWORD)v4 )
      {
        v6 = v4 - 1;
        goto LABEL_9;
      }
LABEL_79:
      ++*(_QWORD *)a1;
      LODWORD(v4) = 0;
      v91 = 0;
      goto LABEL_80;
    }
    v12 = 0;
    v85 = *(_QWORD *)(v3 + 24);
    if ( v85 )
      goto LABEL_7;
    v83 = v3 - 16;
    v18 = *(_BYTE *)(v3 - 16);
    v19 = (v18 & 2) != 0;
    if ( (v18 & 2) != 0 )
      v20 = *(__int64 **)(v3 - 32);
    else
      v20 = (__int64 *)(v83 - 8LL * ((v18 >> 2) & 0xF));
    v21 = *v20;
    if ( *v20 && (_DWORD)v4 )
    {
      a2 = ((_DWORD)v4 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v22 = (__int64 *)(v5 + 16 * a2);
      v23 = *v22;
      if ( *v22 == v21 )
      {
LABEL_20:
        if ( v22 != (__int64 *)(16 * v4 + v5) )
          v21 = v22[1];
      }
      else
      {
        v66 = 1;
        while ( v23 != -4096 )
        {
          v67 = v66 + 1;
          a2 = ((_DWORD)v4 - 1) & (unsigned int)(v66 + a2);
          v22 = (__int64 *)(v5 + 16LL * (unsigned int)a2);
          v23 = *v22;
          if ( v21 == *v22 )
            goto LABEL_20;
          v66 = v67;
        }
      }
    }
    v24 = v20[10];
    if ( v24 )
    {
      v24 = sub_B91420(v20[10], a2);
      v18 = *(_BYTE *)(v3 - 16);
      v81 = v25;
      v19 = (v18 & 2) != 0;
    }
    else
    {
      v81 = 0;
    }
    if ( v19 )
    {
      v88 = *(_QWORD *)(*(_QWORD *)(v3 - 32) + 72LL);
      if ( !v88 )
      {
        v70 = 0;
        v76 = *(unsigned __int8 *)(v3 + 43);
        v75 = *(_DWORD *)(v3 + 36);
        v78 = *(unsigned __int8 *)(v3 + 42);
        v74 = *(unsigned __int8 *)(v3 + 41);
        v72 = *(_QWORD *)(v3 + 24);
LABEL_27:
        v27 = *(_QWORD *)(v3 - 32);
        v68 = *(_QWORD *)(v27 + 64);
        v87 = *(_QWORD *)(v27 + 24);
        if ( !v87 )
        {
          v69 = 0;
          v73 = *(_DWORD *)(v3 + 20);
LABEL_29:
          v86 = *(_QWORD *)(*(_QWORD *)(v3 - 32) + 16LL);
          if ( !v86 )
          {
            v77 = 0;
            v71 = *(unsigned __int8 *)(v3 + 40);
            goto LABEL_31;
          }
          goto LABEL_30;
        }
        goto LABEL_28;
      }
    }
    else
    {
      v88 = *(_QWORD *)(v83 - 8LL * ((v18 >> 2) & 0xF) + 72);
      if ( !v88 )
      {
        v70 = 0;
        v76 = *(unsigned __int8 *)(v3 + 43);
        v75 = *(_DWORD *)(v3 + 36);
        v78 = *(unsigned __int8 *)(v3 + 42);
        v74 = *(unsigned __int8 *)(v3 + 41);
        v72 = *(_QWORD *)(v3 + 24);
        goto LABEL_73;
      }
    }
    v88 = sub_B91420(v88, a2);
    v70 = v26;
    v76 = *(unsigned __int8 *)(v3 + 43);
    v75 = *(_DWORD *)(v3 + 36);
    v78 = *(unsigned __int8 *)(v3 + 42);
    v74 = *(unsigned __int8 *)(v3 + 41);
    v72 = *(_QWORD *)(v3 + 24);
    v18 = *(_BYTE *)(v3 - 16);
    if ( (v18 & 2) != 0 )
      goto LABEL_27;
LABEL_73:
    v42 = 8LL * ((v18 >> 2) & 0xF);
    v68 = *(_QWORD *)(v83 - v42 + 64);
    v87 = *(_QWORD *)(v83 - v42 + 24);
    if ( !v87 )
    {
      v69 = 0;
      v73 = *(_DWORD *)(v3 + 20);
      goto LABEL_75;
    }
LABEL_28:
    v87 = sub_B91420(v87, a2);
    v69 = v28;
    v73 = *(_DWORD *)(v3 + 20);
    v18 = *(_BYTE *)(v3 - 16);
    if ( (v18 & 2) != 0 )
      goto LABEL_29;
LABEL_75:
    v86 = *(_QWORD *)(v3 - 8LL * ((v18 >> 2) & 0xF));
    if ( !v86 )
    {
      v77 = 0;
      v71 = *(unsigned __int8 *)(v3 + 40);
      goto LABEL_77;
    }
LABEL_30:
    v86 = sub_B91420(v86, a2);
    v77 = v29;
    v71 = *(unsigned __int8 *)(v3 + 40);
    v18 = *(_BYTE *)(v3 - 16);
    if ( (v18 & 2) != 0 )
    {
LABEL_31:
      v30 = *(_QWORD *)(v3 - 32);
LABEL_32:
      v31 = *(_QWORD *)(v30 + 8);
      v80 = v31;
      if ( v31 )
      {
        v80 = sub_B91420(v31, a2);
        v85 = v32;
      }
      v79 = *(_DWORD *)(v3 + 16);
      v4 = *(_QWORD *)(v3 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)(v3 + 8) & 4) != 0 )
        v4 = *(_QWORD *)v4;
      v84 = 0;
      if ( v81 )
        v84 = sub_B9B140(v4, v24, v81);
      v82 = 0;
      if ( v70 )
        v82 = sub_B9B140(v4, v88, v70);
      v33 = 0;
      if ( v69 )
        v33 = sub_B9B140(v4, v87, v69);
      v34 = 0;
      if ( v77 )
        v34 = sub_B9B140(v4, v86, v77);
      v35 = 0;
      if ( v85 )
      {
        v89 = v34;
        v36 = sub_B9B140(v4, v80, v85);
        v34 = v89;
        v35 = v36;
      }
      v37 = sub_AF30C0(v4, v79, v21, v35, v71, v34, v73, v33, 2, 0, 0, 0, 0, v68, v72, v74, v78, v75, v76, v82, v84, 1);
      v5 = *(_QWORD *)(a1 + 8);
      LODWORD(v4) = *(_DWORD *)(a1 + 24);
      v12 = v37;
      goto LABEL_7;
    }
LABEL_77:
    v30 = v83 - 8LL * ((v18 >> 2) & 0xF);
    goto LABEL_32;
  }
  v6 = v4 - 1;
  result = ((_DWORD)v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = *(_QWORD *)(v5 + 16 * result);
  if ( a2 != v8 )
  {
    a2 = 1;
    while ( v8 != -4096 )
    {
      result = v6 & (unsigned int)(a2 + result);
      v8 = *(_QWORD *)(v5 + 16LL * (unsigned int)result);
      if ( v3 == v8 )
        return result;
      a2 = (unsigned int)(a2 + 1);
    }
    if ( !v3 )
    {
      v90 = 0;
      v12 = 0;
LABEL_9:
      v13 = 1;
      v14 = 0;
      v15 = v6 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v16 = (__int64 *)(v5 + 16LL * v15);
      v17 = *v16;
      if ( v3 == *v16 )
      {
LABEL_10:
        result = (unsigned __int64)(v16 + 1);
LABEL_11:
        *(_QWORD *)result = v12;
        return result;
      }
      while ( v17 != -4096 )
      {
        if ( !v14 && v17 == -8192 )
          v14 = v16;
        v15 = v6 & (v13 + v15);
        v16 = (__int64 *)(v5 + 16LL * v15);
        v17 = *v16;
        if ( v3 == *v16 )
          goto LABEL_10;
        ++v13;
      }
      if ( !v14 )
        v14 = v16;
      v38 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v39 = v38 + 1;
      v91 = v14;
      if ( 4 * v39 < (unsigned int)(3 * v4) )
      {
        if ( (int)v4 - (v39 + *(_DWORD *)(a1 + 20)) > (unsigned int)v4 >> 3 )
        {
LABEL_61:
          *(_DWORD *)(a1 + 16) = v39;
          if ( *v14 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v14 = v3;
          result = (unsigned __int64)(v14 + 1);
          v14[1] = 0;
          goto LABEL_11;
        }
        v43 = v4;
LABEL_81:
        sub_AEB980(a1, v43);
        sub_AEA890(a1, &v90, &v91);
        v3 = v90;
        v14 = v91;
        v39 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_61;
      }
LABEL_80:
      v43 = 2 * v4;
      goto LABEL_81;
    }
    goto LABEL_5;
  }
  return result;
}
