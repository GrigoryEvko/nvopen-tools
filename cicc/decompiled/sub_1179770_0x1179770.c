// Function: sub_1179770
// Address: 0x1179770
//
__int64 __fastcall sub_1179770(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  int v5; // edx
  __int64 result; // rax
  unsigned int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 i; // r12
  __int64 v21; // rdx
  unsigned int v22; // esi
  unsigned __int8 *v23; // r13
  unsigned __int64 v24; // rax
  __int64 v25; // rcx
  int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // r14
  unsigned int v29; // r15d
  bool v30; // al
  bool v31; // r13
  _QWORD *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // r15
  _BYTE *v39; // rax
  unsigned __int8 *v40; // rdx
  unsigned int v41; // r15d
  int v42; // eax
  unsigned int v43; // ecx
  char v44; // r15
  __int64 v45; // rax
  unsigned int v46; // ecx
  unsigned int v47; // r15d
  unsigned int v48; // [rsp+14h] [rbp-C4h]
  unsigned __int8 *v49; // [rsp+18h] [rbp-C0h]
  __int64 v50; // [rsp+20h] [rbp-B8h]
  __int64 v51; // [rsp+20h] [rbp-B8h]
  __int64 v52; // [rsp+20h] [rbp-B8h]
  int v53; // [rsp+20h] [rbp-B8h]
  __int64 v54; // [rsp+28h] [rbp-B0h] BYREF
  unsigned __int8 *v55; // [rsp+30h] [rbp-A8h] BYREF
  unsigned __int64 v56; // [rsp+3Ch] [rbp-9Ch]
  int v57; // [rsp+44h] [rbp-94h]
  unsigned int v58; // [rsp+48h] [rbp-90h] BYREF
  int v59; // [rsp+4Ch] [rbp-8Ch]
  __int16 v60; // [rsp+68h] [rbp-70h]
  _BYTE v61[32]; // [rsp+78h] [rbp-60h] BYREF
  __int16 v62; // [rsp+98h] [rbp-40h]

  v56 = sub_99AC20(a1, a2, a3, &v54, (__int64 *)&v55, 0);
  v57 = v5;
  if ( (unsigned int)(v56 - 7) > 1 )
  {
    result = 0;
    if ( (_DWORD)v56 )
    {
      v7 = sub_990550(v56);
      v8 = *(_QWORD *)(a4 + 32);
      v59 = 0;
      v62 = 257;
      return sub_B33C40(v8, v7, v54, (__int64)v55, v58, (__int64)v61);
    }
    return result;
  }
  v9 = *(_QWORD *)(a1 + 16);
  if ( !v9 || *(_QWORD *)(v9 + 8) )
  {
    result = *((_QWORD *)v55 + 2);
    if ( !result )
      return result;
    if ( *(_QWORD *)(result + 8) )
      return 0;
  }
  if ( (_DWORD)v56 == 7 )
  {
    v23 = v55;
    v24 = *v55;
    if ( (unsigned __int8)v24 <= 0x1Cu )
    {
      if ( (_BYTE)v24 != 5 )
        goto LABEL_30;
      v26 = *((unsigned __int16 *)v55 + 1);
      if ( (*((_WORD *)v55 + 1) & 0xFFFD) != 0xD && (v26 & 0xFFF7) != 0x11 )
        goto LABEL_30;
    }
    else
    {
      if ( (unsigned __int8)v24 > 0x36u )
        goto LABEL_30;
      v25 = 0x40540000000000LL;
      if ( !_bittest64(&v25, v24) )
        goto LABEL_30;
      v26 = (unsigned __int8)v24 - 29;
    }
    if ( v26 == 15 && (v55[1] & 4) != 0 )
    {
      v27 = *((_QWORD *)v55 - 8);
      v28 = v54;
      if ( *(_BYTE *)v27 == 17 )
      {
        v29 = *(_DWORD *)(v27 + 32);
        if ( v29 <= 0x40 )
          v30 = *(_QWORD *)(v27 + 24) == 0;
        else
          v30 = v29 == (unsigned int)sub_C444A0(v27 + 24);
        if ( v30 )
          goto LABEL_25;
      }
      else
      {
        v38 = *(_QWORD *)(v27 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17 > 1 || *(_BYTE *)v27 > 0x15u )
          goto LABEL_30;
        v52 = *((_QWORD *)v55 - 8);
        v39 = sub_AD7630(v52, 0, v27);
        v40 = (unsigned __int8 *)v52;
        if ( v39 && *v39 == 17 )
        {
          v41 = *((_DWORD *)v39 + 8);
          if ( v41 <= 0x40 )
          {
            if ( *((_QWORD *)v39 + 3) )
              goto LABEL_30;
          }
          else if ( v41 != (unsigned int)sub_C444A0((__int64)(v39 + 24)) )
          {
            goto LABEL_30;
          }
LABEL_25:
          v31 = *((_QWORD *)v23 - 4) == v28;
          v32 = (_QWORD *)sub_BD5C60(a1);
          v33 = v31;
          v34 = sub_BCB2A0(v32);
LABEL_26:
          v35 = sub_ACD640(v34, v33, 0);
          v36 = *(_QWORD *)(a4 + 32);
          v59 = 0;
          v62 = 257;
          return sub_B33C40(v36, 1u, v54, v35, v58, (__int64)v61);
        }
        if ( *(_BYTE *)(v38 + 8) == 17 )
        {
          v42 = *(_DWORD *)(v38 + 32);
          v43 = 0;
          v44 = 0;
          v53 = v42;
          while ( v53 != v43 )
          {
            v48 = v43;
            v49 = v40;
            v45 = sub_AD69F0(v40, v43);
            if ( !v45 )
              goto LABEL_30;
            v40 = v49;
            v46 = v48;
            if ( *(_BYTE *)v45 != 13 )
            {
              if ( *(_BYTE *)v45 != 17 )
                goto LABEL_30;
              v47 = *(_DWORD *)(v45 + 32);
              if ( v47 <= 0x40 )
              {
                if ( *(_QWORD *)(v45 + 24) )
                  goto LABEL_30;
                v44 = 1;
              }
              else
              {
                if ( v47 != (unsigned int)sub_C444A0(v45 + 24) )
                  goto LABEL_30;
                v40 = v49;
                v46 = v48;
                v44 = 1;
              }
            }
            v43 = v46 + 1;
          }
          if ( v44 )
            goto LABEL_25;
        }
      }
    }
LABEL_30:
    v37 = (_QWORD *)sub_BD5C60(a1);
    v33 = 0;
    v34 = sub_BCB2A0(v37);
    goto LABEL_26;
  }
  v10 = (_QWORD *)sub_BD5C60(a1);
  v11 = sub_BCB2A0(v10);
  v12 = sub_ACD640(v11, 0, 0);
  v13 = *(_QWORD *)(a4 + 32);
  v62 = 257;
  v59 = 0;
  v14 = sub_B33C40(v13, 1u, v54, v12, v58, (__int64)v61);
  v15 = *(__int64 **)(a4 + 32);
  v60 = 257;
  v16 = v14;
  v17 = sub_AD6530(*(_QWORD *)(v14 + 8), 1);
  result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v15[10] + 32LL))(
             v15[10],
             15,
             v17,
             v16,
             0,
             0);
  if ( !result )
  {
    v62 = 257;
    v50 = sub_B504D0(15, v17, v16, (__int64)v61, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, unsigned int *, __int64, __int64))(*(_QWORD *)v15[11] + 16LL))(
      v15[11],
      v50,
      &v58,
      v15[7],
      v15[8]);
    result = v50;
    v18 = 16LL * *((unsigned int *)v15 + 2);
    v19 = *v15;
    for ( i = v19 + v18; i != v19; result = v51 )
    {
      v21 = *(_QWORD *)(v19 + 8);
      v22 = *(_DWORD *)v19;
      v19 += 16;
      v51 = result;
      sub_B99FD0(result, v22, v21);
    }
  }
  return result;
}
