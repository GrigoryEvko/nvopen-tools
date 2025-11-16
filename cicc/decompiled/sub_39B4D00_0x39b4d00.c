// Function: sub_39B4D00
// Address: 0x39b4d00
//
__int64 __fastcall sub_39B4D00(__int64 *a1, unsigned int a2, _QWORD *a3, _QWORD *a4, __int64 a5)
{
  __int64 *v5; // r15
  __int64 v8; // r13
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  int v11; // r14d
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  int v15; // r9d
  __int64 v16; // r11
  unsigned int v17; // eax
  __int64 (*v18)(); // rax
  _QWORD *v19; // rdx
  char v20; // al
  char v21; // dl
  int v22; // r13d
  int v23; // r12d
  __int64 *v24; // r13
  int v25; // r14d
  int v26; // r15d
  __int64 v27; // rdx
  int v29; // eax
  __int64 v30; // rsi
  __int64 (*v31)(); // rax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int8 v34; // al
  __int64 v35; // rsi
  __int64 (*v36)(); // rax
  char v37; // al
  int v38; // eax
  int v39; // ecx
  unsigned int v40; // eax
  __int64 v41; // rdx
  unsigned int v42; // eax
  __int64 v43; // rdx
  char v44; // al
  char v45; // al
  _QWORD *v46; // rdx
  int v47; // r13d
  __int64 v48; // rcx
  int i; // r12d
  __int64 v50; // rdx
  int v51; // eax
  __int64 v52; // rcx
  __int64 v53; // rdx
  int v54; // r14d
  _QWORD *v55; // r15
  int v56; // r12d
  __int64 v57; // rdx
  unsigned __int8 v58; // [rsp+3h] [rbp-9Dh]
  unsigned __int8 v59; // [rsp+4h] [rbp-9Ch]
  int v60; // [rsp+8h] [rbp-98h]
  unsigned __int8 v61; // [rsp+8h] [rbp-98h]
  unsigned __int8 v62; // [rsp+8h] [rbp-98h]
  unsigned __int8 v63; // [rsp+8h] [rbp-98h]
  unsigned int v65; // [rsp+18h] [rbp-88h]
  int v66; // [rsp+1Ch] [rbp-84h]
  int v67; // [rsp+20h] [rbp-80h]
  unsigned int v68; // [rsp+24h] [rbp-7Ch]
  unsigned __int8 v69; // [rsp+28h] [rbp-78h]
  __int64 v70; // [rsp+28h] [rbp-78h]
  int v71; // [rsp+28h] [rbp-78h]
  int v72; // [rsp+28h] [rbp-78h]
  int v73; // [rsp+28h] [rbp-78h]
  unsigned __int64 v74; // [rsp+30h] [rbp-70h]
  __int64 v75; // [rsp+30h] [rbp-70h]
  unsigned int v76; // [rsp+38h] [rbp-68h]
  int v77; // [rsp+3Ch] [rbp-64h]
  unsigned __int64 v78; // [rsp+40h] [rbp-60h]
  int v79; // [rsp+40h] [rbp-60h]
  int v81; // [rsp+48h] [rbp-58h]
  int v82; // [rsp+48h] [rbp-58h]
  _BYTE v83[80]; // [rsp+50h] [rbp-50h] BYREF

  v5 = a1;
  v77 = 1;
  v59 = 4 * (a2 == 37) + 8;
  v68 = 0;
  v65 = a2 - 37;
  while ( 1 )
  {
    v8 = a1[2];
    v76 = sub_1F43D70(v8, a2);
    v10 = sub_1F43D80(v8, *a1, (__int64)a4, v9);
    v67 = v10;
    v11 = v10;
    v74 = HIDWORD(v10);
    v69 = BYTE4(v10);
    v13 = sub_1F43D80(v8, *a1, (__int64)a3, v12);
    v66 = v13;
    v15 = v13;
    v78 = HIDWORD(v13);
    v16 = BYTE4(v13);
    if ( (_DWORD)v13 != v11 )
      goto LABEL_94;
    sub_39B1510(v74);
    v29 = sub_39B1510(v78);
    if ( (_DWORD)v14 == v29 )
    {
      if ( a2 == 47 || a2 == 36 )
        return v68;
      v17 = a2;
      if ( a2 == 37 )
        goto LABEL_27;
    }
    else
    {
LABEL_94:
      if ( a2 == 36 )
      {
        v35 = v69;
        v14 = (unsigned __int8)v16;
        v36 = *(__int64 (**)())(*(_QWORD *)v8 + 800LL);
        if ( v36 != sub_1D12DF0 )
        {
          v61 = v16;
          v71 = v15;
          v37 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, _QWORD))v36)(
                  v8,
                  v35,
                  0,
                  (unsigned __int8)v16,
                  0);
          v15 = v71;
          v16 = v61;
          if ( v37 )
            return v68;
        }
        goto LABEL_11;
      }
      v17 = a2;
      if ( a2 == 37 )
      {
LABEL_27:
        v30 = v69;
        v14 = (unsigned __int8)v16;
        v31 = *(__int64 (**)())(*(_QWORD *)v8 + 824LL);
        if ( v31 != sub_1D12E00 )
        {
          v62 = v16;
          v72 = v15;
          v44 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, _QWORD))v31)(
                  v8,
                  v30,
                  0,
                  (unsigned __int8)v16,
                  0);
          v15 = v72;
          v16 = v62;
          if ( v44 )
            return v68;
        }
        goto LABEL_28;
      }
    }
    if ( v17 == 48 )
    {
      v14 = (__int64)a3;
      v18 = *(__int64 (**)())(*(_QWORD *)v8 + 576LL);
      if ( *((_BYTE *)a3 + 8) == 16 )
        v14 = *(_QWORD *)a3[2];
      v19 = a4;
      if ( *((_BYTE *)a4 + 8) == 16 )
        v19 = *(_QWORD **)a4[2];
      if ( v18 != sub_1D12D90 )
      {
        v73 = v15;
        v63 = v16;
        v45 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v18)(
                v8,
                *((_DWORD *)v19 + 2) >> 8,
                *(_DWORD *)(v14 + 8) >> 8);
        v15 = v73;
        v16 = v63;
        if ( v45 )
          return v68;
      }
      goto LABEL_11;
    }
LABEL_28:
    if ( v65 <= 1 && a5 )
    {
      if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
      {
        v32 = *(_QWORD *)(a5 - 8);
      }
      else
      {
        v14 = a5;
        v32 = a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF);
      }
      if ( *(_BYTE *)(*(_QWORD *)v32 + 16LL) == 54 )
      {
        v58 = v16;
        v60 = v15;
        LOBYTE(v33) = sub_1F59570((__int64)a3);
        v70 = v33;
        v34 = sub_1F59570((__int64)a4);
        v14 = v70;
        v15 = v60;
        v16 = v58;
        if ( (_BYTE)v70 )
        {
          if ( v34 )
          {
            v14 = v59;
            if ( (((int)*(unsigned __int16 *)(v8 + 2 * (v34 + 115LL * (unsigned __int8)v70 + 16104)) >> v59) & 0xF) == 0 )
              return v68;
            if ( v60 != v11 )
              goto LABEL_12;
            goto LABEL_37;
          }
        }
      }
    }
LABEL_11:
    if ( v15 != v11 )
      goto LABEL_12;
LABEL_37:
    if ( ((_BYTE)v78 == 1 || (_BYTE)v78 && *(_QWORD *)(v8 + 8LL * (unsigned __int8)v78 + 120)) && v76 <= 0x102 )
    {
      v14 = 129LL * (unsigned __int8)v78;
      if ( *(_BYTE *)(v76 + v8 + 259LL * (unsigned __int8)v78 + 2422) <= 1u )
        goto LABEL_42;
    }
LABEL_12:
    v20 = *((_BYTE *)a4 + 8);
    v21 = *((_BYTE *)a3 + 8);
    if ( v20 != 16 )
      break;
    if ( v21 != 16 )
    {
      v22 = 0;
      if ( (int)a4[4] > 0 )
      {
        v54 = 0;
        v55 = a4;
        v56 = a4[4];
        while ( 1 )
        {
          v57 = (__int64)v55;
          if ( v20 == 16 )
            v57 = *(_QWORD *)v55[2];
          ++v54;
          v22 += sub_1F43D80(a1[2], *a1, v57, v14);
          if ( v56 == v54 )
            break;
          v20 = *((_BYTE *)v55 + 8);
        }
        v5 = a1;
        if ( *((_BYTE *)a3 + 8) != 16 )
        {
          v68 += v77 * v22;
          return v68;
        }
LABEL_15:
        if ( (int)a3[4] > 0 )
        {
          v81 = v22;
          v23 = 0;
          v24 = v5;
          v25 = 0;
          v26 = a3[4];
          do
          {
            v27 = (__int64)a3;
            if ( *((_BYTE *)a3 + 8) == 16 )
              v27 = *(_QWORD *)a3[2];
            ++v25;
            v23 += sub_1F43D80(v24[2], *v24, v27, v14);
          }
          while ( v26 != v25 );
          v22 = v23 + v81;
        }
      }
      v68 += v77 * v22;
      return v68;
    }
    if ( v15 == v11 )
    {
      sub_39B1510(v74);
      v38 = sub_39B1510(v78);
      if ( v39 == v38 )
      {
        if ( a2 == 37 )
          goto LABEL_42;
        if ( a2 == 38 )
        {
          v68 += 2 * v77;
          return v68;
        }
        if ( (_BYTE)v78 && *(_QWORD *)(v8 + 8LL * (unsigned __int8)v78 + 120) )
        {
          if ( v76 > 0x102 )
          {
            v68 += v77 * v66;
            return v68;
          }
          if ( *(_BYTE *)(v76 + v8 + 259LL * (unsigned __int8)v78 + 2422) != 2 )
          {
            v68 += v77 * v67;
            return v68;
          }
        }
      }
    }
    LOBYTE(v40) = sub_39B1E70(*a1, (__int64)a4);
    sub_1F40D10((__int64)v83, v8, *a4, v40, v41);
    if ( v83[0] != 6 )
    {
      LOBYTE(v42) = sub_39B1E70(*a1, (__int64)a3);
      sub_1F40D10((__int64)v83, v8, *a3, v42, v43);
      if ( v83[0] != 6 )
      {
        v75 = a3[4];
        if ( *((_BYTE *)a4 + 8) == 16 )
          a4 = *(_QWORD **)a4[2];
        v46 = a3;
        if ( *((_BYTE *)a3 + 8) == 16 )
          v46 = *(_QWORD **)a3[2];
        v47 = 0;
        v79 = sub_39B4D00(a1, a2, v46, a4, a5);
        v82 = a3[4];
        if ( v82 > 0 )
        {
          for ( i = 0; i != v82; ++i )
          {
            v50 = (__int64)a3;
            if ( *((_BYTE *)a3 + 8) == 16 )
              v50 = *(_QWORD *)a3[2];
            v51 = sub_1F43D80(a1[2], *a1, v50, v48);
            v53 = (__int64)a3;
            if ( *((_BYTE *)a3 + 8) == 16 )
              v53 = *(_QWORD *)a3[2];
            v47 += sub_1F43D80(a1[2], *a1, v53, v52) + v51;
          }
        }
        v68 += v77 * (v75 * v79 + v47);
        return v68;
      }
    }
    a3 = sub_16463B0(*(__int64 **)a3[2], *((_DWORD *)a3 + 8) >> 1);
    a4 = sub_16463B0(*(__int64 **)a4[2], *((_DWORD *)a4 + 8) >> 1);
    v68 += v77;
    v77 *= 2;
  }
  if ( v21 == 16 )
  {
    v22 = 0;
    goto LABEL_15;
  }
  if ( a2 != 47 )
  {
    if ( (_BYTE)v16
      && *(_QWORD *)(v8 + 8 * v16 + 120)
      && (v76 > 0x102 || *(_BYTE *)(v76 + 259LL * (unsigned __int8)v16 + v8 + 2422) != 2) )
    {
LABEL_42:
      v68 += v77;
    }
    else
    {
      v68 += 4 * v77;
    }
  }
  return v68;
}
