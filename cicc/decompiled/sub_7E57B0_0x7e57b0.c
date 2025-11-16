// Function: sub_7E57B0
// Address: 0x7e57b0
//
__int64 __fastcall sub_7E57B0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _WORD *a7)
{
  __int64 v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // r13
  __int64 v11; // r14
  int v12; // r15d
  int v13; // eax
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 *v16; // rdx
  __int64 **v17; // rax
  __int64 *v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 **v21; // rax
  __int64 v22; // rdi
  unsigned __int16 v23; // si
  __int64 result; // rax
  __int64 v25; // rbx
  __int64 v26; // r15
  __int64 v27; // rbx
  __int64 *v28; // rbx
  __int64 v29; // rdi
  _QWORD *v30; // rax
  __int64 v31; // rax
  bool v32; // zf
  __int64 v33; // r15
  __int64 v34; // rdi
  _BYTE *v35; // r14
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // r15
  __int64 v39; // r11
  __int64 v40; // r10
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // [rsp+8h] [rbp-98h]
  _QWORD *v49; // [rsp+10h] [rbp-90h]
  __int64 v50; // [rsp+10h] [rbp-90h]
  __int64 v51; // [rsp+10h] [rbp-90h]
  __int64 v52; // [rsp+10h] [rbp-90h]
  __int64 v53; // [rsp+10h] [rbp-90h]
  __int64 v57; // [rsp+30h] [rbp-70h]
  __int64 v58; // [rsp+38h] [rbp-68h]
  unsigned __int16 v61; // [rsp+56h] [rbp-4Ah] BYREF
  __int64 v62; // [rsp+58h] [rbp-48h] BYREF
  __int64 v63; // [rsp+60h] [rbp-40h] BYREF
  __int64 v64[7]; // [rsp+68h] [rbp-38h] BYREF

  if ( a4 )
  {
    v8 = *(_QWORD *)(a4 + 40);
    v9 = *(_QWORD **)(a4 + 120);
    v61 = 0;
    v10 = *(_QWORD *)(v8 + 168);
    v58 = v8;
    v11 = *(_QWORD *)(v10 + 24);
    if ( v11 )
    {
      v12 = a6;
      v13 = sub_8E5310(*(_QWORD *)(v10 + 24), a3, a4);
      sub_7E57B0((_DWORD)a1, (_DWORD)a2, a3, v13, a5, v12, (__int64)&v61);
      for ( ; v9; v9 = (_QWORD *)*v9 )
      {
        if ( *(_WORD *)(v9[2] + 224LL) >= v61 )
          break;
      }
      v14 = *(_QWORD *)(*(_QWORD *)(v58 + 168) + 16LL);
      if ( v14 )
        goto LABEL_8;
LABEL_39:
      v26 = sub_7E0220(a4);
      if ( !v26 )
        goto LABEL_30;
      v27 = 0;
      if ( a6 )
        v27 = *(_QWORD *)(a6 + 104);
      if ( !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v26 + 40) + 168LL) + 64LL) )
        goto LABEL_30;
      v57 = v27;
      v28 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v26 + 40) + 168LL) + 64LL);
      while ( 1 )
      {
        while ( *((_BYTE *)v28 + 32) )
        {
          v28 = (__int64 *)*v28;
          if ( !v28 )
            goto LABEL_30;
        }
        v29 = v26;
        if ( v28[2] )
          v29 = sub_8E5650(v28[2]);
        v30 = *(_QWORD **)(v29 + 120);
        if ( v30 )
        {
          while ( v30[2] != v28[1] )
          {
            v30 = (_QWORD *)*v30;
            if ( !v30 )
              goto LABEL_99;
          }
          v29 = v30[3];
          if ( !a5 )
            goto LABEL_101;
          if ( v29 )
            goto LABEL_100;
          v29 = a5;
        }
        else
        {
LABEL_99:
          if ( a5 )
          {
LABEL_100:
            v29 = sub_8E5650(v29);
LABEL_101:
            v31 = 0;
            if ( !v29 )
              goto LABEL_56;
          }
        }
        v31 = *(_QWORD *)(v29 + 104);
LABEL_56:
        sub_7E2E30(v31 - v57, 0, 0, a1, a2, 1, *(_QWORD *)(v26 + 40));
        v28 = (__int64 *)*v28;
        if ( !v28 )
          goto LABEL_30;
      }
    }
    v14 = *(_QWORD *)(v10 + 16);
    if ( !v14 )
      goto LABEL_39;
  }
  else
  {
    v10 = *(_QWORD *)(a3 + 168);
    v61 = 0;
    v11 = *(_QWORD *)(v10 + 24);
    if ( v11 )
    {
      v9 = 0;
      sub_7E57B0((_DWORD)a1, (_DWORD)a2, a3, v11, a5, a6, (__int64)&v61);
      v58 = a3;
      v14 = *(_QWORD *)(*(_QWORD *)(a3 + 168) + 16LL);
      if ( !v14 )
        goto LABEL_30;
    }
    else
    {
      v9 = *(_QWORD **)(v10 + 16);
      v58 = a3;
      if ( !v9 )
        goto LABEL_30;
      v14 = *(_QWORD *)(v10 + 16);
      v9 = 0;
    }
  }
LABEL_8:
  v49 = v9;
  v15 = v14;
  do
  {
    if ( (*(_BYTE *)(v15 + 96) & 2) != 0 )
    {
      v16 = *(__int64 **)(v15 + 40);
      if ( v11 && (v17 = **(__int64 ****)(*(_QWORD *)(v11 + 40) + 168LL)) != 0 )
      {
        while ( 1 )
        {
          if ( ((_BYTE)v17[12] & 2) != 0 )
          {
            v18 = v17[5];
            if ( v18 == v16 )
              break;
            if ( v18 )
            {
              if ( v16 )
              {
                if ( dword_4F07588 )
                {
                  v19 = v16[4];
                  if ( v18[4] == v19 )
                  {
                    if ( v19 )
                      break;
                  }
                }
              }
            }
          }
          v17 = (__int64 **)*v17;
          if ( !v17 )
            goto LABEL_20;
        }
      }
      else
      {
LABEL_20:
        v20 = a3;
        if ( a5 )
          v20 = *(_QWORD *)(a5 + 56);
        v21 = *(__int64 ***)(v20 + 168);
        do
        {
          do
            v21 = (__int64 **)*v21;
          while ( v16 != v21[5] );
        }
        while ( ((_BYTE)v21[12] & 2) == 0 );
        v22 = (__int64)v21[13];
        if ( a6 )
          v22 -= *(_QWORD *)(a6 + 104);
        v62 = v22;
        sub_7E2E30(v22, 0, 0, a1, a2, 1, v58);
      }
    }
    v15 = *(_QWORD *)(v15 + 16);
  }
  while ( v15 );
  v9 = v49;
  if ( a4 )
    goto LABEL_39;
LABEL_30:
  v23 = *(_WORD *)(v10 + 44);
  result = v61;
  if ( v23 != 0xFFFF && v23 >= v61 )
  {
    v25 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v25 )
          v25 = *(_QWORD *)(*(_QWORD *)(v10 + 152) + 144LL);
        if ( (*(_BYTE *)(v25 + 192) & 2) != 0 && *(_WORD *)(v25 + 224) == (_WORD)result )
          break;
        v25 = *(_QWORD *)(v25 + 112);
      }
      if ( v9 && v9[2] == v25 )
        break;
      v32 = *(_BYTE *)(v25 + 174) == 2;
      v62 = 0;
      if ( v32 )
      {
        v35 = (_BYTE *)sub_7FDF40(v25, 3, 0);
        v44 = sub_7FDF40(v25, 1, 0);
        v34 = v62;
        v33 = v44;
LABEL_93:
        if ( !HIDWORD(qword_4F077B4) )
          goto LABEL_69;
        goto LABEL_63;
      }
      if ( !HIDWORD(qword_4F077B4) )
      {
        sub_7E2E30(0, v25, 0, a1, a2, 0, v58);
        v37 = v61;
        goto LABEL_71;
      }
      v33 = v25;
      v34 = 0;
      v35 = 0;
LABEL_63:
      if ( qword_4F077A8 > 0x9FC3u
        && !(_DWORD)qword_4F077B4
        && a5
        && (*(_BYTE *)(v33 + 174) == 2 && (*(_BYTE *)(v33 + 192) & 2) != 0
         || (v36 = *(_QWORD *)(v33 + 272)) != 0 && *(_BYTE *)(v36 + 174) == 2) )
      {
        sub_7E2E30(v34, 0, 0, a1, a2, 0, v58);
        if ( !v35 )
        {
LABEL_74:
          v37 = v61;
          goto LABEL_71;
        }
        v35 = 0;
        goto LABEL_70;
      }
LABEL_69:
      sub_7E2E30(v34, v33, 0, a1, a2, 0, v58);
      if ( !v35 )
        goto LABEL_74;
LABEL_70:
      sub_7E2E30(v62, (__int64)v35, 0, a1, a2, 0, v58);
      v37 = v61 + 1;
LABEL_71:
      result = (unsigned int)(v37 + 1);
      v25 = *(_QWORD *)(v25 + 112);
      v61 = result;
      if ( v23 < (unsigned __int16)result )
        goto LABEL_72;
    }
    v38 = v9[3];
    v39 = a4;
    if ( a5 )
    {
      v38 = v38 ? sub_8E5650(v9[3]) : a5;
      v39 = a5;
      if ( a4 )
        v39 = sub_8E5650(a4);
    }
    v40 = v9[1];
    v35 = 0;
    if ( *(_BYTE *)(v40 + 174) == 2 )
    {
      v48 = v39;
      v52 = v9[1];
      v35 = (_BYTE *)sub_7FDF40(v52, 3, 0);
      v45 = sub_7FDF40(v52, 1, 0);
      v39 = v48;
      v40 = v45;
    }
    v41 = v9[4];
    v50 = v40;
    v63 = 0;
    sub_7E02A0((__int64 **)v25, v38, v39, a6, v41, 1, v40, &v62, v64, &v63);
    v42 = v9[4];
    if ( v42 )
    {
      v34 = v62;
      if ( !(*(_QWORD *)(v42 + 104) | v62)
        && !v64[0]
        && (*(_BYTE *)(v42 + 96) & 2) == 0
        && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v42 + 112) + 8LL) + 16LL) + 96LL) & 2) == 0 )
      {
LABEL_106:
        v33 = v50;
LABEL_92:
        v9 = (_QWORD *)*v9;
        goto LABEL_93;
      }
    }
    else if ( !v62 )
    {
      v34 = v64[0];
      if ( !v64[0] )
        goto LABEL_106;
    }
    v33 = v63;
    if ( !v63 )
    {
      v33 = v50;
      if ( (*(_BYTE *)(v50 + 192) & 8) == 0 && (*(_BYTE *)(v50 + 206) & 0x10) == 0 )
      {
        v46 = v50;
        v53 = v9[4];
        v47 = sub_7E5350(v46, v9[2], v53, v62, v64[0]);
        v42 = v53;
        v33 = v47;
      }
    }
    v51 = v42;
    if ( v35 )
    {
      v43 = sub_7FDF40(v9[2], (v35[205] >> 2) & 7, 0);
      if ( (v35[192] & 8) == 0 && (v35[206] & 0x10) == 0 )
        v35 = (_BYTE *)sub_7E5350((__int64)v35, v43, v51, v62, v64[0]);
    }
    v62 = 0;
    v34 = 0;
    goto LABEL_92;
  }
LABEL_72:
  *a7 = result;
  return result;
}
