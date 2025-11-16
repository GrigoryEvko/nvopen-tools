// Function: sub_1596970
// Address: 0x1596970
//
__int64 __fastcall sub_1596970(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rax
  __int64 *v4; // rbx
  unsigned __int64 v5; // r13
  __int64 *v6; // rax
  int v7; // edx
  __int64 *v8; // rsi
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // rdx
  __int64 v12; // rbx
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r13
  unsigned int v18; // eax
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // r13
  _QWORD *v22; // r15
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // r14
  _QWORD *v28; // r15
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // r14
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // r14
  _QWORD *v38; // r13
  __int64 v39; // r15
  __int64 v40; // rax
  unsigned int v41; // ebx
  bool v42; // zf
  __int64 v43; // r14
  __int64 **v44; // r13
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r11
  __int64 **v50; // rcx
  __int64 **v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r13
  __int64 v54; // r14
  __int64 v55; // r15
  __int64 v56; // rax
  __int64 v57; // r15
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // r11
  __int64 **v62; // rcx
  __int64 **v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned int v67; // [rsp+4h] [rbp-9Ch]
  unsigned int v68; // [rsp+4h] [rbp-9Ch]
  __int64 v69; // [rsp+8h] [rbp-98h]
  __int64 v70; // [rsp+8h] [rbp-98h]
  __int64 v71; // [rsp+10h] [rbp-90h]
  __int64 v72; // [rsp+18h] [rbp-88h]
  _QWORD *v73; // [rsp+18h] [rbp-88h]
  unsigned int v74; // [rsp+18h] [rbp-88h]
  _BYTE v75[16]; // [rsp+20h] [rbp-80h] BYREF
  __int16 v76; // [rsp+30h] [rbp-70h]
  __int64 *v77; // [rsp+40h] [rbp-60h] BYREF
  __int64 v78; // [rsp+48h] [rbp-58h]
  _BYTE v79[80]; // [rsp+50h] [rbp-50h] BYREF

  v2 = *(_DWORD *)(a1 + 20);
  v77 = (__int64 *)v79;
  v78 = 0x400000000LL;
  v3 = 24LL * (v2 & 0xFFFFFFF);
  v4 = (__int64 *)(a1 - v3);
  v5 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
  if ( (unsigned __int64)v3 > 0x60 )
  {
    sub_16CD150(&v77, v79, 0xAAAAAAAAAAAAAAABLL * (v3 >> 3), 8);
    v8 = v77;
    v7 = v78;
    v6 = &v77[(unsigned int)v78];
  }
  else
  {
    v6 = (__int64 *)v79;
    v7 = 0;
    v8 = (__int64 *)v79;
  }
  if ( (__int64 *)a1 != v4 )
  {
    do
    {
      if ( v6 )
        *v6 = *v4;
      v4 += 3;
      ++v6;
    }
    while ( (__int64 *)a1 != v4 );
    v8 = v77;
    v7 = v78;
  }
  v9 = *(unsigned __int16 *)(a1 + 18);
  v10 = v7 + v5;
  LODWORD(v78) = v10;
  switch ( (__int16)v9 )
  {
    case ' ':
      v41 = v10;
      v42 = (*(_BYTE *)(a1 + 17) & 2) == 0;
      v76 = 257;
      v43 = v10 - 1LL;
      v44 = (__int64 **)(v8 + 1);
      v74 = v41;
      v71 = *v8;
      if ( v42 )
      {
        v57 = sub_16348C0(a1);
        if ( !v57 )
        {
          v65 = *(_QWORD *)v71;
          if ( *(_BYTE *)(*(_QWORD *)v71 + 8LL) == 16 )
            v65 = **(_QWORD **)(v65 + 16);
          v57 = *(_QWORD *)(v65 + 24);
        }
        v58 = sub_1648A60(72, v41);
        v12 = v58;
        if ( v58 )
        {
          v70 = v58 - 24LL * v74;
          v59 = *(_QWORD *)v71;
          if ( *(_BYTE *)(*(_QWORD *)v71 + 8LL) == 16 )
            v59 = **(_QWORD **)(v59 + 16);
          v68 = *(_DWORD *)(v59 + 8) >> 8;
          v60 = sub_15F9F50(v57, v44, v43);
          v61 = sub_1646BA0(v60, v68);
          if ( *(_BYTE *)(*(_QWORD *)v71 + 8LL) == 16 )
          {
            v61 = sub_16463B0(v61, *(_QWORD *)(*(_QWORD *)v71 + 32LL));
          }
          else
          {
            v62 = &v44[v43];
            if ( v62 != v44 )
            {
              v63 = (__int64 **)(v8 + 1);
              while ( 1 )
              {
                v64 = **v63;
                if ( *(_BYTE *)(v64 + 8) == 16 )
                  break;
                if ( v62 == ++v63 )
                  goto LABEL_56;
              }
              v61 = sub_16463B0(v61, *(_QWORD *)(v64 + 32));
            }
          }
LABEL_56:
          sub_15F1EA0(v12, v61, 32, v70, v74, 0);
          *(_QWORD *)(v12 + 56) = v57;
          *(_QWORD *)(v12 + 64) = sub_15F9F50(v57, v44, v43);
          sub_15F9CE0(v12, v71, v44, v43, v75);
        }
      }
      else
      {
        v45 = sub_16348C0(a1);
        if ( !v45 )
        {
          v66 = *(_QWORD *)v71;
          if ( *(_BYTE *)(*(_QWORD *)v71 + 8LL) == 16 )
            v66 = **(_QWORD **)(v66 + 16);
          v45 = *(_QWORD *)(v66 + 24);
        }
        v46 = sub_1648A60(72, v41);
        v12 = v46;
        if ( v46 )
        {
          v69 = v46 - 24LL * v74;
          v47 = *(_QWORD *)v71;
          if ( *(_BYTE *)(*(_QWORD *)v71 + 8LL) == 16 )
            v47 = **(_QWORD **)(v47 + 16);
          v67 = *(_DWORD *)(v47 + 8) >> 8;
          v48 = sub_15F9F50(v45, v44, v43);
          v49 = sub_1646BA0(v48, v67);
          if ( *(_BYTE *)(*(_QWORD *)v71 + 8LL) == 16 )
          {
            v49 = sub_16463B0(v49, *(_QWORD *)(*(_QWORD *)v71 + 32LL));
          }
          else
          {
            v50 = &v44[v43];
            if ( v50 != v44 )
            {
              v51 = (__int64 **)(v8 + 1);
              while ( 1 )
              {
                v52 = **v51;
                if ( *(_BYTE *)(v52 + 8) == 16 )
                  break;
                if ( v50 == ++v51 )
                  goto LABEL_42;
              }
              v49 = sub_16463B0(v49, *(_QWORD *)(v52 + 32));
            }
          }
LABEL_42:
          sub_15F1EA0(v12, v49, 32, v69, v74, 0);
          *(_QWORD *)(v12 + 56) = v45;
          *(_QWORD *)(v12 + 64) = sub_15F9F50(v45, v44, v43);
          sub_15F9CE0(v12, v71, v44, v43, v75);
        }
        sub_15FA2E0(v12, 1);
      }
      break;
    case '$':
    case '%':
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case '0':
      v11 = *(_QWORD *)a1;
      v76 = 257;
      v12 = sub_15FDBD0(v9, *v8, v11, v75, 0);
      break;
    case '3':
    case '4':
      v76 = 257;
      v16 = v8[1];
      v17 = *v8;
      v18 = sub_1594720(a1);
      v12 = sub_15FEEB0(*(unsigned __int16 *)(a1 + 18), v18, v17, v16, v75, 0);
      break;
    case '7':
      v76 = 257;
      v37 = v8[2];
      v38 = (_QWORD *)v8[1];
      v39 = *v8;
      v40 = sub_1648A60(56, 3);
      v12 = v40;
      if ( v40 )
      {
        v73 = (_QWORD *)(v40 - 72);
        sub_15F1EA0(v40, *v38, 55, v40 - 72, 3, 0);
        sub_1593B40(v73, v39);
        sub_1593B40((_QWORD *)(v12 - 48), (__int64)v38);
        sub_1593B40((_QWORD *)(v12 - 24), v37);
        sub_164B780(v12, v75);
      }
      break;
    case ';':
      v76 = 257;
      v30 = v8[1];
      v31 = *v8;
      v32 = sub_1648A60(56, 2);
      v12 = v32;
      if ( v32 )
        sub_15FA320(v32, v31, v30, v75, 0);
      break;
    case '<':
      v76 = 257;
      v33 = v8[2];
      v34 = v8[1];
      v35 = *v8;
      v36 = sub_1648A60(56, 3);
      v12 = v36;
      if ( v36 )
        sub_15FA480(v36, v35, v34, v33, v75, 0);
      break;
    case '=':
      v53 = *v8;
      v54 = v8[1];
      v55 = v8[2];
      v76 = 257;
      v56 = sub_1648A60(56, 3);
      v12 = v56;
      if ( v56 )
        sub_15FA660(v56, v53, v54, v55, v75, 0);
      break;
    case '>':
      v76 = 257;
      v19 = sub_1594710(a1);
      v21 = v20;
      v22 = (_QWORD *)*v8;
      v12 = sub_1648A60(88, 1);
      if ( v12 )
      {
        v23 = sub_15FB2A0(*v22, v19, v21);
        sub_15F1EA0(v12, v23, 62, v12 - 24, 1, 0);
        sub_1593B40((_QWORD *)(v12 - 24), (__int64)v22);
        *(_QWORD *)(v12 + 56) = v12 + 72;
        *(_QWORD *)(v12 + 64) = 0x400000000LL;
        sub_15FB110(v12, v19, v21, v75);
      }
      break;
    case '?':
      v76 = 257;
      v24 = sub_1594710(a1);
      v72 = v25;
      v26 = v24;
      v27 = v8[1];
      v28 = (_QWORD *)*v8;
      v29 = sub_1648A60(88, 2);
      v12 = v29;
      if ( v29 )
      {
        sub_15F1EA0(v29, *v28, 63, v29 - 48, 2, 0);
        *(_QWORD *)(v12 + 56) = v12 + 72;
        *(_QWORD *)(v12 + 64) = 0x400000000LL;
        sub_15FAD90(v12, v28, v27, v26, v72, v75);
      }
      break;
    default:
      v76 = 257;
      v12 = sub_15FB440(v9, *v8, v8[1], v75, 0);
      v14 = *(unsigned __int8 *)(v12 + 16);
      if ( (unsigned __int8)v14 <= 0x2Fu )
      {
        v15 = 0x80A800000000LL;
        if ( _bittest64(&v15, v14) )
        {
          sub_15F2310(v12, (*(_BYTE *)(a1 + 17) & 2) != 0);
          sub_15F2330(v12, (*(_BYTE *)(a1 + 17) & 4) != 0);
          LODWORD(v14) = *(unsigned __int8 *)(v12 + 16);
        }
      }
      if ( (unsigned __int8)(v14 - 48) <= 1u || (unsigned int)(v14 - 41) <= 1 )
        sub_15F2350(v12, (*(_BYTE *)(a1 + 17) & 2) != 0);
      break;
  }
  if ( v77 != (__int64 *)v79 )
    _libc_free((unsigned __int64)v77);
  return v12;
}
