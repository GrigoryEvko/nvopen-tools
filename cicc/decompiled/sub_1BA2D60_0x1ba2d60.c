// Function: sub_1BA2D60
// Address: 0x1ba2d60
//
__int64 __fastcall sub_1BA2D60(__int64 a1, __int64 **a2, unsigned int a3, __int64 **a4)
{
  unsigned int v5; // r12d
  __int64 *v7; // r14
  char v8; // al
  __int64 *v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r8d
  __int64 v17; // r10
  __int64 v18; // rdi
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // r10
  int v24; // esi
  unsigned __int64 v25; // rdx
  __int64 **v26; // rcx
  unsigned __int64 v27; // r9
  __int64 *v28; // rsi
  unsigned int v29; // r14d
  __int64 **v30; // rax
  __int64 v31; // rdi
  __int64 v32; // r15
  unsigned int v34; // r14d
  __int64 *v35; // rdi
  __int64 *v36; // rax
  __int64 v37; // rsi
  __int64 ***v38; // rax
  __int64 **v39; // rsi
  __int64 *v40; // r14
  unsigned int v41; // eax
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // rdi
  __int64 v46; // rbx
  __int64 *v47; // r14
  __int64 v48; // r8
  __int64 *v49; // r15
  char v50; // al
  __int64 *v51; // rax
  __int64 v52; // r14
  int v53; // ebx
  _QWORD *v54; // rax
  __int64 *v55; // rax
  int v56; // eax
  __int64 *v57; // rax
  __int64 v58; // r14
  _QWORD *v59; // rax
  __int64 *v60; // rax
  _QWORD *v61; // rax
  int v62; // r14d
  int v63; // ebx
  int v64; // ebx
  __int64 *v65; // rax
  __int64 v66; // [rsp+0h] [rbp-90h]
  _QWORD *v67; // [rsp+8h] [rbp-88h]
  unsigned __int64 v68; // [rsp+10h] [rbp-80h]
  __int64 v69; // [rsp+18h] [rbp-78h]
  __int64 v70; // [rsp+28h] [rbp-68h] BYREF
  __int64 **v71; // [rsp+30h] [rbp-60h] BYREF
  __int64 v72; // [rsp+38h] [rbp-58h]
  _BYTE v73[80]; // [rsp+40h] [rbp-50h] BYREF

  v5 = a3;
  v70 = (__int64)a2;
  v7 = *a2;
  if ( (unsigned __int8)sub_1B96020(a1, (__int64)a2, a3) )
  {
    v12 = (_QWORD *)sub_14C5C70(a1 + 8, (unsigned __int64 *)&v70);
    v7 = (__int64 *)sub_1644900((_QWORD *)*v7, *v12);
  }
  v8 = sub_1B95F70(a1, v70, v5);
  v9 = v7;
  if ( !v8 )
    v9 = sub_1B8E090(v7, v5);
  v10 = v70;
  *a4 = v9;
  v11 = *(unsigned __int8 *)(v10 + 16);
  switch ( *(_BYTE *)(v10 + 16) )
  {
    case 0x1A:
      if ( v5 <= 1 )
      {
        v46 = *(_QWORD *)(v10 + 40);
        if ( v46 != sub_13FCB50(*(_QWORD *)(a1 + 296)) && (v5 & 1) == 0 )
          return 0;
        return (unsigned int)sub_14A3410(*(_QWORD *)(a1 + 328));
      }
      else
      {
        if ( (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 3 )
          goto LABEL_56;
        if ( !sub_183E920(a1 + 64, *(_QWORD *)(v10 - 24)) && !sub_183E920(a1 + 64, *(_QWORD *)(v10 - 48)) )
        {
          v10 = v70;
LABEL_56:
          v44 = *(_QWORD *)(v10 + 40);
          if ( v44 == sub_13FCB50(*(_QWORD *)(a1 + 296)) )
            return (unsigned int)sub_14A3410(*(_QWORD *)(a1 + 328));
          return 0;
        }
        v60 = (__int64 *)sub_1643320((_QWORD *)*v7);
        v61 = sub_16463B0(v60, v5);
        v62 = sub_14A2E40(*(__int64 **)(a1 + 328), (__int64)v61, 0, 1u);
        return (unsigned int)sub_14A3410(*(_QWORD *)(a1 + 328)) * v5 + v62;
      }
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x2B:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
      goto LABEL_7;
    case 0x29:
    case 0x2A:
    case 0x2C:
    case 0x2D:
      if ( v5 <= 1 )
        goto LABEL_7;
      if ( (unsigned __int8)sub_1B91FD0(a1, v10) )
      {
        v63 = sub_14A3410(*(_QWORD *)(a1 + 328));
        v64 = v5 * (sub_14A3350(*(_QWORD *)(a1 + 328)) + v63);
        return (v64 + (unsigned int)sub_1B8FA60(v70, v5, *(__int64 **)(a1 + 328))) >> 1;
      }
      v10 = v70;
      v11 = *(unsigned __int8 *)(v70 + 16);
LABEL_7:
      if ( (_BYTE)v11 == 39 )
      {
        v57 = (__int64 *)sub_13CF970(v10);
        v58 = *(_QWORD *)(a1 + 320);
        if ( sub_1A018F0(*(_QWORD *)(v58 + 48) + 96LL, *v57) )
          return 0;
        v11 = *(_QWORD *)(sub_13CF970(v10) + 24);
        if ( sub_1A018F0(*(_QWORD *)(v58 + 48) + 96LL, v11) )
          return 0;
        v10 = v70;
      }
      v13 = sub_13CF970(v10);
      v17 = v10;
      v18 = *(_QWORD *)(v13 + 24);
      v19 = *(_BYTE *)(v18 + 16);
      if ( v19 == 13 )
      {
        if ( *(_DWORD *)(v18 + 32) <= 0x40u )
          goto LABEL_13;
        v69 = v10;
        v45 = v18 + 24;
        goto LABEL_63;
      }
      if ( (v19 & 0xFB) == 8 )
      {
        v20 = sub_15A1020((_BYTE *)v18, v11, v14, v15);
        v10 = v70;
        v17 = v70;
        if ( !v20 || *(_BYTE *)(v20 + 16) != 13 || *(_DWORD *)(v20 + 32) <= 0x40u )
          goto LABEL_13;
        v69 = v70;
        v45 = v20 + 24;
LABEL_63:
        sub_16A5940(v45);
        v17 = v69;
        goto LABEL_13;
      }
      sub_1BF2190(*(_QWORD *)(a1 + 320), v18);
      v10 = v70;
      v17 = v70;
LABEL_13:
      v21 = 3LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
      v22 = (_QWORD *)(v10 - v21 * 8);
      if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      {
        v22 = *(_QWORD **)(v10 - 8);
        v17 = (__int64)&v22[v21];
      }
      v23 = v17 - (_QWORD)v22;
      v24 = 0;
      v71 = (__int64 **)v73;
      v72 = 0x400000000LL;
      v25 = 0xAAAAAAAAAAAAAAABLL * (v23 >> 3);
      v26 = (__int64 **)v73;
      v27 = v25;
      if ( (unsigned __int64)v23 > 0x60 )
      {
        v68 = 0xAAAAAAAAAAAAAAABLL * (v23 >> 3);
        v66 = v23;
        v67 = v22;
        sub_16CD150((__int64)&v71, v73, v25, 8, v16, v25);
        v24 = v72;
        v23 = v66;
        v22 = v67;
        v27 = v68;
        LODWORD(v25) = v68;
        v26 = &v71[(unsigned int)v72];
      }
      if ( v23 > 0 )
      {
        do
        {
          v28 = (__int64 *)*v22;
          ++v26;
          v22 += 3;
          *(v26 - 1) = v28;
          --v27;
        }
        while ( v27 );
        v24 = v72;
      }
      LODWORD(v72) = v24 + v25;
      if ( !sub_1B95F70(a1, v70, v5) )
        v5 = 1;
      v29 = v5 * sub_14A3350(*(_QWORD *)(a1 + 328));
      if ( v71 != (__int64 **)v73 )
        _libc_free((unsigned __int64)v71);
      return v29;
    case 0x36:
    case 0x37:
      v34 = v5;
      if ( v5 > 1 )
      {
        v56 = sub_1B99570(a1, v10, v5);
        v10 = v70;
        LOBYTE(v11) = *(_BYTE *)(v70 + 16);
        if ( v56 == 5 )
          v34 = 1;
      }
      if ( (_BYTE)v11 == 54 )
        v35 = *(__int64 **)v10;
      else
        v35 = **(__int64 ***)(v10 - 48);
      v36 = sub_1B8E090(v35, v34);
      v37 = v70;
      *a4 = v36;
      return (unsigned int)sub_1BA2B80(a1, v37, v5);
    case 0x38:
      return 0;
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
      if ( (_BYTE)v11 != 60 )
        goto LABEL_72;
      v30 = *(__int64 ***)(v10 - 24);
      if ( *((_BYTE *)*v30 + 8) && v5 != 1 )
        sub_16463B0(*v30, v5);
      sub_1B8E090(*(__int64 **)v10, v5);
      v31 = *(_QWORD *)(a1 + 320);
      v32 = *(_QWORD *)(v10 - 24);
      if ( v32 == *(_QWORD *)(v31 + 64) )
        goto LABEL_31;
      if ( !(unsigned __int8)sub_14A2CF0(*(_QWORD *)(a1 + 328)) )
      {
        v31 = *(_QWORD *)(a1 + 320);
LABEL_31:
        if ( (unsigned __int8)sub_1BF2700(v31, v32) )
          return (unsigned int)sub_14A33B0(*(_QWORD *)(a1 + 328));
      }
      v10 = v70;
LABEL_72:
      v47 = **(__int64 ***)sub_13CF970(v10);
      if ( *((_BYTE *)*a4 + 8) == 16 )
      {
        sub_1B8E090(v47, v5);
        v10 = v70;
      }
      if ( !(unsigned __int8)sub_1B96020(a1, v10, v5) )
        goto LABEL_80;
      v48 = v70;
      v49 = *a4;
      v50 = *(_BYTE *)(v70 + 16);
      if ( v50 == 60 )
      {
        v65 = sub_1B8E090(*(__int64 **)v70, v5);
        v48 = v70;
        if ( *(_DWORD *)(*(_QWORD *)v65[2] + 8LL) >> 8 <= *(_DWORD *)(*(_QWORD *)v49[2] + 8LL) >> 8 )
          v65 = v49;
        *a4 = v65;
      }
      else if ( (unsigned __int8)(v50 - 61) <= 1u )
      {
        v51 = sub_1B8E090(*(__int64 **)v70, v5);
        if ( *(_DWORD *)(*(_QWORD *)v51[2] + 8LL) >> 8 >= *(_DWORD *)(*(_QWORD *)v49[2] + 8LL) >> 8 )
          v51 = v49;
        *a4 = v51;
LABEL_80:
        v48 = v70;
      }
      if ( !sub_1B95F70(a1, v48, v5) )
        v5 = 1;
      return v5 * (unsigned int)sub_14A33B0(*(_QWORD *)(a1 + 328));
    case 0x4B:
    case 0x4C:
      v38 = (__int64 ***)sub_13CF970(v10);
      v39 = *v38;
      v40 = **v38;
      if ( *((_BYTE *)*v38 + 16) <= 0x17u )
        v39 = 0;
      v71 = v39;
      if ( (unsigned __int8)sub_1B96020(a1, (__int64)v39, v5) )
      {
        v59 = (_QWORD *)sub_14C5C70(a1 + 8, (unsigned __int64 *)&v71);
        v40 = (__int64 *)sub_1644900((_QWORD *)*v40, *v59);
      }
      *a4 = sub_1B8E090(v40, v5);
      return (unsigned int)sub_14A3440(*(_QWORD *)(a1 + 328));
    case 0x4D:
      if ( v5 <= 1 )
        return (unsigned int)sub_14A3410(*(_QWORD *)(a1 + 328));
      if ( (unsigned __int8)sub_1BF28F0(*(_QWORD *)(a1 + 320), v10) )
      {
        return (unsigned int)sub_14A3380(*(_QWORD *)(a1 + 328));
      }
      else
      {
        if ( *(_QWORD *)(v10 + 40) == **(_QWORD **)(*(_QWORD *)(a1 + 296) + 32LL) )
          return (unsigned int)sub_14A3410(*(_QWORD *)(a1 + 328));
        v52 = *(_QWORD *)(a1 + 328);
        v53 = *(_DWORD *)(v10 + 20);
        v54 = (_QWORD *)sub_16498A0(v10);
        v55 = (__int64 *)sub_1643320(v54);
        sub_1B8E090(v55, v5);
        sub_1B8E090(*(__int64 **)v10, v5);
        return ((v53 & 0xFFFFFFF) - 1) * (unsigned int)sub_14A3440(v52);
      }
    case 0x4E:
      v29 = sub_1B8FD60(v10, v5, *(__int64 **)(a1 + 328), *(__int64 **)(a1 + 336), &v71);
      if ( (unsigned int)sub_14C3B40(v10, *(__int64 **)(a1 + 336)) )
      {
        v41 = sub_1B8F4A0(v10, v5, *(_QWORD *)(a1 + 328), *(__int64 **)(a1 + 336));
        if ( v29 > v41 )
          return v41;
      }
      return v29;
    case 0x4F:
      v42 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 112LL);
      v43 = sub_146F1B0(v42, *(_QWORD *)(v10 - 72));
      if ( !sub_146CEE0(v42, v43, *(_QWORD *)(a1 + 296)) )
        sub_16463B0(**(__int64 ***)(v10 - 72), v5);
      return (unsigned int)sub_14A3440(*(_QWORD *)(a1 + 328));
    default:
      v29 = v5 * sub_14A3350(*(_QWORD *)(a1 + 328));
      if ( v5 != 1 )
        v29 += sub_1B8FA60(v70, v5, *(__int64 **)(a1 + 328));
      return v29;
  }
}
