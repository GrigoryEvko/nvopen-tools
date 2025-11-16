// Function: sub_279BEA0
// Address: 0x279bea0
//
__int64 __fastcall sub_279BEA0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdi
  _BYTE *v5; // rdi
  unsigned int v6; // r13d
  __int64 v7; // r9
  unsigned __int64 v8; // rdi
  __int64 v10; // rcx
  unsigned __int8 **v11; // r12
  unsigned __int8 **v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  int v15; // ecx
  __int64 v16; // rdi
  int v17; // ecx
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // r11
  char v21; // al
  __int64 v22; // r9
  unsigned int v23; // r8d
  char v24; // al
  __int64 v25; // r13
  unsigned __int8 v26; // al
  __int64 v27; // rsi
  __int64 v28; // rcx
  int v29; // edx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 *v33; // r15
  __int64 v34; // r12
  __int64 v35; // rax
  unsigned __int64 *v36; // r13
  unsigned __int64 *v37; // r12
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned __int8 *v42; // rsi
  int v43; // eax
  int v44; // r9d
  unsigned __int8 v45; // [rsp+Fh] [rbp-1451h]
  __int64 v46; // [rsp+30h] [rbp-1430h] BYREF
  __int64 v47; // [rsp+38h] [rbp-1428h] BYREF
  _QWORD v48[2]; // [rsp+40h] [rbp-1420h] BYREF
  __int64 v49[10]; // [rsp+50h] [rbp-1410h] BYREF
  unsigned __int64 *v50; // [rsp+A0h] [rbp-13C0h]
  unsigned int v51; // [rsp+A8h] [rbp-13B8h]
  char v52; // [rsp+B0h] [rbp-13B0h] BYREF
  _BYTE *v53; // [rsp+200h] [rbp-1260h] BYREF
  __int64 v54; // [rsp+208h] [rbp-1258h]
  _BYTE v55[512]; // [rsp+210h] [rbp-1250h] BYREF
  _BYTE *v56; // [rsp+410h] [rbp-1050h] BYREF
  __int64 v57; // [rsp+418h] [rbp-1048h]
  _BYTE v58[1536]; // [rsp+420h] [rbp-1040h] BYREF
  __int64 *v59; // [rsp+A20h] [rbp-A40h] BYREF
  __int64 v60; // [rsp+A28h] [rbp-A38h]
  _BYTE v61[2608]; // [rsp+A30h] [rbp-A30h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v57 = 0x4000000000LL;
  v56 = v58;
  sub_1036F30(v4, a2, (__int64)&v56, 1);
  if ( (unsigned int)v57 > (unsigned int)qword_4FFBB28 )
  {
    v5 = v56;
    v6 = 0;
    goto LABEL_14;
  }
  if ( (_DWORD)v57 != 1 || (v5 = v56, v6 = 0, (*((_DWORD *)v56 + 2) & 7u) - 1 <= 1) )
  {
    if ( (unsigned __int8)sub_278A900((unsigned __int8 *)a1)
      && (unsigned __int8)sub_278A920(a1)
      && (v10 = *(_QWORD *)(a2 - 32), *(_BYTE *)v10 == 63)
      && v10 != v10 + 32 * (1LL - (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)) )
    {
      v6 = 0;
      v11 = *(unsigned __int8 ***)(a2 - 32);
      v12 = (unsigned __int8 **)(v10 + 32 * (1LL - (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)));
      do
      {
        if ( **v12 > 0x1Cu )
          v6 |= sub_27987E0(a1, *v12);
        v12 += 4;
      }
      while ( v11 != v12 );
      v3 = a2;
    }
    else
    {
      v6 = 0;
    }
    v59 = (__int64 *)v61;
    v53 = v55;
    v60 = 0x4000000000LL;
    v54 = 0x4000000000LL;
    sub_278DD60(a1, v3, (__int64)&v56, (__int64)&v59, (__int64)&v53, v7);
    if ( (_DWORD)v60 )
    {
      if ( !(_DWORD)v54 )
      {
        v25 = sub_278B590(v3, &v59, a1);
        sub_30EC4B0(*(_QWORD *)(a1 + 104), v3);
        sub_BD84D0(v3, v25);
        v26 = *(_BYTE *)v25;
        if ( *(_BYTE *)v25 == 84 )
        {
          sub_102BD20(*(_QWORD *)(a1 + 16), v25);
          sub_BD6B90((unsigned __int8 *)v25, (unsigned __int8 *)v3);
          v26 = *(_BYTE *)v25;
        }
        if ( v26 > 0x1Cu )
        {
          v27 = *(_QWORD *)(v3 + 48);
          if ( v27 )
          {
            if ( *(_QWORD *)(v3 + 40) == *(_QWORD *)(v25 + 40) )
            {
              v49[0] = *(_QWORD *)(v3 + 48);
              sub_B96E90((__int64)v49, v27, 1);
              if ( (__int64 *)(v25 + 48) == v49 )
              {
                if ( v49[0] )
                  sub_B91220(v25 + 48, v49[0]);
              }
              else
              {
                v41 = *(_QWORD *)(v25 + 48);
                if ( v41 )
                  sub_B91220(v25 + 48, v41);
                v42 = (unsigned __int8 *)v49[0];
                *(_QWORD *)(v25 + 48) = v49[0];
                if ( v42 )
                  sub_B976B0((__int64)v49, v42, v25 + 48);
              }
            }
          }
        }
        v28 = *(_QWORD *)(v25 + 8);
        v29 = *(unsigned __int8 *)(v28 + 8);
        if ( (unsigned int)(v29 - 17) <= 1 )
          LOBYTE(v29) = *(_BYTE *)(**(_QWORD **)(v28 + 16) + 8LL);
        if ( (_BYTE)v29 == 14 )
          sub_102B9D0(*(_QWORD *)(a1 + 16), v25);
        sub_278A7A0(a1 + 136, (_BYTE *)v3);
        v32 = *(unsigned int *)(a1 + 656);
        if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 660) )
        {
          sub_C8D5F0(a1 + 648, (const void *)(a1 + 664), v32 + 1, 8u, v30, v31);
          v32 = *(unsigned int *)(a1 + 656);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8 * v32) = v3;
        ++*(_DWORD *)(a1 + 656);
        v33 = *(__int64 **)(a1 + 96);
        v48[0] = &v47;
        v47 = v3;
        v46 = v25;
        v48[1] = &v46;
        v34 = *v33;
        v35 = sub_B2BE50(*v33);
        if ( sub_B6EA50(v35)
          || (v39 = sub_B2BE50(v34),
              v40 = sub_B6F970(v39),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v40 + 48LL))(v40)) )
        {
          sub_2790350((__int64)v49, (__int64)v48);
          sub_1049740(v33, (__int64)v49);
          v36 = v50;
          v49[0] = (__int64)&unk_49D9D40;
          v37 = &v50[10 * v51];
          if ( v50 != v37 )
          {
            do
            {
              v37 -= 10;
              v38 = v37[4];
              if ( (unsigned __int64 *)v38 != v37 + 6 )
                j_j___libc_free_0(v38);
              if ( (unsigned __int64 *)*v37 != v37 + 2 )
                j_j___libc_free_0(*v37);
            }
            while ( v36 != v37 );
            v37 = v50;
          }
          if ( v37 != (unsigned __int64 *)&v52 )
            _libc_free((unsigned __int64)v37);
        }
        v8 = (unsigned __int64)v53;
        v6 = 1;
        if ( v53 == v55 )
          goto LABEL_11;
        goto LABEL_10;
      }
      if ( (unsigned __int8)sub_278A900((unsigned __int8 *)a1) )
      {
        v45 = sub_278A940(a1);
        if ( v45 )
        {
          if ( !(unsigned __int8)sub_278A960(a1) )
          {
            v13 = *(_QWORD *)(a1 + 112);
            v14 = *(_QWORD *)(v3 + 40);
            v15 = *(_DWORD *)(v13 + 24);
            v16 = *(_QWORD *)(v13 + 8);
            if ( v15 )
            {
              v17 = v15 - 1;
              v18 = v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
              v19 = (__int64 *)(v16 + 16LL * v18);
              v20 = *v19;
              if ( v14 == *v19 )
              {
LABEL_30:
                if ( v19[1] )
                  goto LABEL_9;
              }
              else
              {
                v43 = 1;
                while ( v20 != -4096 )
                {
                  v44 = v43 + 1;
                  v18 = v17 & (v43 + v18);
                  v19 = (__int64 *)(v16 + 16LL * v18);
                  v20 = *v19;
                  if ( v14 == *v19 )
                    goto LABEL_30;
                  v43 = v44;
                }
              }
            }
          }
          v21 = sub_279BB20((__int64 *)a1, v3, (__int64)&v59, (__int64)&v53);
          v23 = v45;
          if ( v21 || (v24 = sub_2799EE0((__int64 *)a1, v3, (__int64 *)&v59, (__int64)&v53, v45, v22), v23 = v45, v24) )
            v6 = v23;
        }
      }
    }
LABEL_9:
    v8 = (unsigned __int64)v53;
    if ( v53 == v55 )
    {
LABEL_11:
      if ( v59 != (__int64 *)v61 )
        _libc_free((unsigned __int64)v59);
      v5 = v56;
      goto LABEL_14;
    }
LABEL_10:
    _libc_free(v8);
    goto LABEL_11;
  }
LABEL_14:
  if ( v5 != v58 )
    _libc_free((unsigned __int64)v5);
  return v6;
}
