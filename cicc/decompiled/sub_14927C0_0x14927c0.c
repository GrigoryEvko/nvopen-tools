// Function: sub_14927C0
// Address: 0x14927c0
//
__int64 *__fastcall sub_14927C0(__int64 *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 *v4; // r15
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 **v9; // rdx
  __int64 *v10; // r10
  int v12; // edx
  __int64 v13; // rdx
  _QWORD *v14; // rbx
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rdx
  _QWORD *v27; // rbx
  __int64 *v28; // rax
  __int64 v29; // r12
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 *v35; // rbx
  __int64 *v36; // r13
  __int64 v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rdx
  _QWORD *v42; // rbx
  __int64 v43; // rax
  int v44; // r11d
  __int64 v45; // rax
  __int64 v46; // rbx
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // r15
  __int64 v51; // r13
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rbx
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rdi
  __int64 v59; // rdi
  _QWORD *v60; // [rsp+8h] [rbp-A8h]
  _QWORD *v61; // [rsp+8h] [rbp-A8h]
  _QWORD *v62; // [rsp+8h] [rbp-A8h]
  _QWORD *v63; // [rsp+8h] [rbp-A8h]
  _QWORD *v64; // [rsp+8h] [rbp-A8h]
  char v65; // [rsp+1Ch] [rbp-94h]
  char v66; // [rsp+1Ch] [rbp-94h]
  char v67; // [rsp+1Ch] [rbp-94h]
  char v68; // [rsp+1Ch] [rbp-94h]
  char v69; // [rsp+1Ch] [rbp-94h]
  unsigned int v70; // [rsp+1Ch] [rbp-94h]
  __int64 v71; // [rsp+20h] [rbp-90h]
  __int64 v72; // [rsp+20h] [rbp-90h]
  __int64 v73; // [rsp+20h] [rbp-90h]
  __int64 v74; // [rsp+20h] [rbp-90h]
  __int64 v75; // [rsp+20h] [rbp-90h]
  __int64 v76; // [rsp+20h] [rbp-90h]
  __int64 v77; // [rsp+28h] [rbp-88h] BYREF
  __int64 *v78; // [rsp+38h] [rbp-78h] BYREF
  __int64 *v79; // [rsp+40h] [rbp-70h] BYREF
  __int64 v80; // [rsp+48h] [rbp-68h]
  __int64 v81; // [rsp+50h] [rbp-60h] BYREF
  char v82; // [rsp+58h] [rbp-58h] BYREF
  char v83; // [rsp+70h] [rbp-40h]

  v4 = (__int64 *)a2;
  v6 = *((unsigned int *)a1 + 8);
  v77 = a2;
  if ( (_DWORD)v6 )
  {
    v7 = a1[2];
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 **)(v7 + 16LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
    {
LABEL_3:
      if ( v9 != (__int64 **)(v7 + 16 * v6) )
        return v9[1];
    }
    else
    {
      v12 = 1;
      while ( v10 != (__int64 *)-8LL )
      {
        v44 = v12 + 1;
        v8 = (v6 - 1) & (v12 + v8);
        v9 = (__int64 **)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v4 == *v9 )
          goto LABEL_3;
        v12 = v44;
      }
    }
  }
  switch ( *((_WORD *)v4 + 12) )
  {
    case 0:
    case 0xB:
      goto LABEL_9;
    case 1:
      v40 = sub_14927C0(a1, v4[4]);
      if ( v40 != v4[4] )
        v4 = (__int64 *)sub_14835F0((_QWORD *)*a1, v40, v4[5], 0, a3, a4);
      goto LABEL_9;
    case 2:
      v25 = sub_14927C0(a1, v4[4]);
      if ( *(_WORD *)(v25 + 24) != 7 || *(_QWORD *)(v25 + 48) != a1[7] || *(_QWORD *)(v25 + 40) != 2 )
        goto LABEL_29;
      v45 = sub_13A5BC0((_QWORD *)v25, *a1);
      v46 = v4[5];
      v47 = v45;
      v48 = sub_145DF90(*a1, v25, 1u);
      v49 = a1[5];
      if ( v49 )
      {
        sub_1412190(v49, v48);
LABEL_73:
        v50 = *a1;
        v70 = *(_WORD *)(v25 + 26) & 7;
        v76 = a1[7];
        v51 = sub_147B0D0(*a1, v47, v46, 0);
        v52 = sub_14747F0(*a1, **(_QWORD **)(v25 + 32), v46, 0);
        goto LABEL_74;
      }
      v58 = a1[6];
      if ( v58 && sub_1454560(v58, v48) )
        goto LABEL_73;
LABEL_29:
      v4 = (__int64 *)sub_14747F0(*a1, v25, v4[5], 0);
LABEL_9:
      v78 = v4;
      sub_1466830((__int64)&v79, (__int64)(a1 + 1), &v77, (__int64 *)&v78);
      return *(__int64 **)(v81 + 8);
    case 3:
      v29 = sub_14927C0(a1, v4[4]);
      if ( *(_WORD *)(v29 + 24) != 7 || *(_QWORD *)(v29 + 48) != a1[7] || *(_QWORD *)(v29 + 40) != 2 )
        goto LABEL_37;
      v53 = sub_13A5BC0((_QWORD *)v29, *a1);
      v54 = v4[5];
      v55 = v53;
      v56 = sub_145DF90(*a1, v29, 2u);
      v57 = a1[5];
      if ( v57 )
      {
        sub_1412190(v57, v56);
      }
      else
      {
        v59 = a1[6];
        if ( !v59 || !sub_1454560(v59, v56) )
        {
LABEL_37:
          v4 = (__int64 *)sub_147B0D0(*a1, v29, v4[5], 0);
          goto LABEL_9;
        }
      }
      v50 = *a1;
      v70 = *(_WORD *)(v29 + 26) & 7;
      v76 = a1[7];
      v51 = sub_147B0D0(*a1, v55, v54, 0);
      v52 = sub_147B0D0(*a1, **(_QWORD **)(v29 + 32), v54, 0);
LABEL_74:
      v4 = (__int64 *)sub_14799E0(v50, v52, v51, v76, v70);
      goto LABEL_9;
    case 4:
      v79 = &v81;
      v80 = 0x200000000LL;
      v26 = v4[4];
      v63 = (_QWORD *)(v26 + 8 * v4[5]);
      if ( (_QWORD *)v26 == v63 )
        goto LABEL_9;
      v68 = 0;
      v27 = (_QWORD *)v4[4];
      do
      {
        v74 = *v27;
        v78 = (__int64 *)sub_14927C0(a1, *v27);
        sub_1458920((__int64)&v79, &v78);
        v15 = v79;
        ++v27;
        v68 |= v79[(unsigned int)v80 - 1] != v74;
      }
      while ( v63 != v27 );
      if ( v68 )
      {
        v28 = sub_147DD40(*a1, (__int64 *)&v79, 0, 0, a3, a4);
        v15 = v79;
        v4 = v28;
      }
      goto LABEL_64;
    case 5:
      v79 = &v81;
      v80 = 0x200000000LL;
      v41 = v4[4];
      v64 = (_QWORD *)(v41 + 8 * v4[5]);
      if ( (_QWORD *)v41 == v64 )
        goto LABEL_9;
      v69 = 0;
      v42 = (_QWORD *)v4[4];
      do
      {
        v75 = *v42;
        v78 = (__int64 *)sub_14927C0(a1, *v42);
        sub_1458920((__int64)&v79, &v78);
        v15 = v79;
        ++v42;
        v69 |= v79[(unsigned int)v80 - 1] != v75;
      }
      while ( v64 != v42 );
      if ( v69 )
      {
        v43 = sub_147EE30((_QWORD *)*a1, &v79, 0, 0, a3, a4);
        v15 = v79;
        v4 = (__int64 *)v43;
      }
      goto LABEL_64;
    case 6:
      v23 = sub_14927C0(a1, v4[4]);
      v24 = sub_14927C0(a1, v4[5]);
      if ( v23 != v4[4] || v24 != v4[5] )
        v4 = (__int64 *)sub_1483CF0((_QWORD *)*a1, v23, v24, a3, a4);
      goto LABEL_9;
    case 7:
      v79 = &v81;
      v80 = 0x200000000LL;
      v20 = v4[4];
      v62 = (_QWORD *)(v20 + 8 * v4[5]);
      if ( (_QWORD *)v20 == v62 )
        goto LABEL_9;
      v67 = 0;
      v21 = (_QWORD *)v4[4];
      do
      {
        v73 = *v21;
        v78 = (__int64 *)sub_14927C0(a1, *v21);
        sub_1458920((__int64)&v79, &v78);
        v15 = v79;
        ++v21;
        v67 |= v79[(unsigned int)v80 - 1] != v73;
      }
      while ( v62 != v21 );
      if ( v67 )
      {
        v22 = sub_14785F0(*a1, &v79, v4[6], *((_WORD *)v4 + 13) & 7);
        v15 = v79;
        v4 = (__int64 *)v22;
      }
      goto LABEL_64;
    case 8:
      v79 = &v81;
      v80 = 0x200000000LL;
      v17 = v4[4];
      v61 = (_QWORD *)(v17 + 8 * v4[5]);
      if ( (_QWORD *)v17 == v61 )
        goto LABEL_9;
      v66 = 0;
      v18 = (_QWORD *)v4[4];
      do
      {
        v72 = *v18;
        v78 = (__int64 *)sub_14927C0(a1, *v18);
        sub_1458920((__int64)&v79, &v78);
        v15 = v79;
        ++v18;
        v66 |= v79[(unsigned int)v80 - 1] != v72;
      }
      while ( v61 != v18 );
      if ( v66 )
      {
        v19 = sub_14813B0((_QWORD *)*a1, &v79, a3, a4);
        v15 = v79;
        v4 = (__int64 *)v19;
      }
      goto LABEL_64;
    case 9:
      v79 = &v81;
      v80 = 0x200000000LL;
      v13 = v4[4];
      v60 = (_QWORD *)(v13 + 8 * v4[5]);
      if ( (_QWORD *)v13 == v60 )
        goto LABEL_9;
      v65 = 0;
      v14 = (_QWORD *)v4[4];
      do
      {
        v71 = *v14;
        v78 = (__int64 *)sub_14927C0(a1, *v14);
        sub_1458920((__int64)&v79, &v78);
        v15 = v79;
        ++v14;
        v65 |= v79[(unsigned int)v80 - 1] != v71;
      }
      while ( v60 != v14 );
      if ( v65 )
      {
        v16 = sub_147A3C0((_QWORD *)*a1, &v79, a3, a4);
        v15 = v79;
        v4 = (__int64 *)v16;
      }
LABEL_64:
      if ( v15 != &v81 )
        goto LABEL_65;
      goto LABEL_9;
    case 0xA:
      v30 = a1[6];
      if ( !v30 )
        goto LABEL_43;
      v31 = sub_1458650(v30, (__int64)v4);
      v33 = v31 + 8 * v32;
      if ( v31 == v33 )
        goto LABEL_43;
      do
      {
        v34 = *(_QWORD *)v31;
        if ( *(_DWORD *)(*(_QWORD *)v31 + 32LL) == 1 && v4 == *(__int64 **)(v34 + 40) )
        {
          v4 = *(__int64 **)(v34 + 48);
          goto LABEL_9;
        }
        v31 += 8;
      }
      while ( v33 != v31 );
LABEL_43:
      if ( *(_BYTE *)(*(v4 - 1) + 16) == 77 )
      {
        sub_1492380((__int64)&v79, *a1, (__int64)(v4 - 4), a3, a4);
        if ( v83 )
        {
          v35 = (__int64 *)v80;
          v36 = (__int64 *)(v80 + 8LL * (unsigned int)v81);
          if ( (__int64 *)v80 == v36 )
          {
LABEL_79:
            v4 = v79;
          }
          else
          {
            while ( 1 )
            {
              v37 = *v35;
              if ( *(_DWORD *)(*v35 + 32) == 2 )
              {
                v37 = *v35;
                if ( a1[7] != *(_QWORD *)(sub_1452540(*v35) + 48) )
                  break;
              }
              v38 = a1[5];
              if ( v38 )
              {
                sub_1412190(v38, v37);
              }
              else
              {
                v39 = a1[6];
                if ( !v39 || !sub_1454560(v39, v37) )
                  break;
              }
              if ( v36 == ++v35 )
                goto LABEL_79;
            }
          }
          if ( v83 )
          {
            v15 = (__int64 *)v80;
            if ( (char *)v80 != &v82 )
LABEL_65:
              _libc_free((unsigned __int64)v15);
          }
        }
      }
      goto LABEL_9;
  }
}
