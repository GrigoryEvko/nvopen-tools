// Function: sub_2465B30
// Address: 0x2465b30
//
_BYTE *__fastcall sub_2465B30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // rdx
  unsigned int v8; // ebx
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  _BYTE *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned __int8 *v19; // r13
  __int64 (__fastcall *v20)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v21; // r15
  _BYTE *v22; // rax
  _QWORD *v23; // rdi
  _BYTE *v24; // rdx
  _BYTE *v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned __int8 *v29; // r9
  __int64 (__fastcall *v30)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  _QWORD *v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  unsigned __int8 *v37; // r13
  __int64 (__fastcall *v38)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v39; // r9
  _QWORD *v41; // rax
  _QWORD *v42; // r9
  __int64 v43; // rdx
  __int64 v44; // r15
  __int64 v45; // r13
  unsigned int *v46; // rbx
  __int64 v47; // rdx
  unsigned int v48; // esi
  _QWORD *v49; // rax
  unsigned int *v50; // rbx
  __int64 v51; // r13
  __int64 v52; // rdx
  unsigned int v53; // esi
  _QWORD *v54; // rax
  __int64 v55; // r15
  unsigned int *v56; // r14
  __int64 v57; // r13
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 *v60; // rax
  __int64 v61; // rax
  unsigned __int8 *v62; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v63; // [rsp+10h] [rbp-E0h]
  __int64 v64; // [rsp+20h] [rbp-D0h]
  __int64 v65; // [rsp+28h] [rbp-C8h]
  char v66; // [rsp+34h] [rbp-BCh]
  _BYTE *v67; // [rsp+38h] [rbp-B8h]
  _BYTE *v69; // [rsp+48h] [rbp-A8h]
  _QWORD *v70; // [rsp+48h] [rbp-A8h]
  __int64 v71; // [rsp+48h] [rbp-A8h]
  __int64 v72; // [rsp+48h] [rbp-A8h]
  __int64 v73; // [rsp+48h] [rbp-A8h]
  _BYTE v76[32]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v77; // [rsp+80h] [rbp-70h]
  _BYTE v78[32]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v79; // [rsp+B0h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
    return sub_24657C0(a1, a2, (unsigned int **)a3, a4, a5);
  v8 = *(_DWORD *)(v7 + 32);
  v9 = (__int64 *)sub_BCE3C0(*(__int64 **)(a3 + 72), 0);
  v10 = sub_BCDA70(v9, v8);
  v67 = 0;
  v11 = sub_AD6530(v10, v8);
  if ( *(_DWORD *)(a1[1] + 4) )
  {
    v60 = (__int64 *)sub_BCE3C0(*(__int64 **)(a3 + 72), 0);
    v61 = sub_BCDA70(v60, v8);
    v67 = (_BYTE *)sub_AD6530(v61, v8);
  }
  if ( v8 )
  {
    v12 = (_BYTE *)v11;
    v66 = a5;
    v13 = v8;
    v14 = 0;
    v65 = v13;
    while ( 1 )
    {
      v15 = *(_QWORD **)(a3 + 72);
      v77 = 257;
      v16 = sub_BCB2D0(v15);
      v17 = sub_ACD640(v16, v14, 0);
      v18 = *(_QWORD *)(a3 + 80);
      v19 = (unsigned __int8 *)v17;
      v20 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v18 + 96LL);
      if ( v20 != sub_948070 )
        break;
      if ( *(_BYTE *)a2 <= 0x15u && *v19 <= 0x15u )
      {
        v21 = sub_AD5840(a2, v19, 0);
        goto LABEL_11;
      }
LABEL_37:
      v79 = 257;
      v49 = sub_BD2C40(72, 2u);
      v21 = (__int64)v49;
      if ( v49 )
        sub_B4DE80((__int64)v49, a2, (__int64)v19, (__int64)v78, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v21,
        v76,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
      {
        v73 = v14;
        v50 = *(unsigned int **)a3;
        v51 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        do
        {
          v52 = *((_QWORD *)v50 + 1);
          v53 = *v50;
          v50 += 4;
          sub_B99FD0(v21, v53, v52);
        }
        while ( (unsigned int *)v51 != v50 );
        v14 = v73;
      }
LABEL_12:
      v22 = sub_24657C0(a1, v21, (unsigned int **)a3, a4, v66);
      v23 = *(_QWORD **)(a3 + 72);
      v69 = v24;
      v25 = v22;
      v77 = 257;
      v26 = sub_BCB2D0(v23);
      v27 = sub_ACD640(v26, v14, 0);
      v28 = *(_QWORD *)(a3 + 80);
      v29 = (unsigned __int8 *)v27;
      v30 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v28 + 104LL);
      if ( v30 == sub_948040 )
      {
        v31 = 0;
        if ( *v12 <= 0x15u )
          v31 = (__int64)v12;
        if ( *v25 > 0x15u || *v29 > 0x15u || !v31 )
        {
LABEL_43:
          v79 = 257;
          v64 = (__int64)v29;
          v54 = sub_BD2C40(72, 3u);
          v55 = (__int64)v54;
          if ( v54 )
            sub_B4DFA0((__int64)v54, (__int64)v12, (__int64)v25, v64, (__int64)v78, v64, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v55,
            v76,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v56 = *(unsigned int **)a3;
          v57 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v57 )
          {
            do
            {
              v58 = *((_QWORD *)v56 + 1);
              v59 = *v56;
              v56 += 4;
              sub_B99FD0(v55, v59, v58);
            }
            while ( (unsigned int *)v57 != v56 );
          }
          v12 = (_BYTE *)v55;
          goto LABEL_21;
        }
        v62 = v29;
        v32 = sub_AD5A90(v31, v25, v29, 0);
        v29 = v62;
      }
      else
      {
        v63 = v29;
        v32 = v30(v28, v12, v25, v29);
        v29 = v63;
      }
      if ( !v32 )
        goto LABEL_43;
      v12 = (_BYTE *)v32;
LABEL_21:
      if ( *(_DWORD *)(a1[1] + 4) )
      {
        v33 = *(_QWORD **)(a3 + 72);
        v77 = 257;
        v34 = sub_BCB2D0(v33);
        v35 = sub_ACD640(v34, v14, 0);
        v36 = *(_QWORD *)(a3 + 80);
        v37 = (unsigned __int8 *)v35;
        v38 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v36 + 104LL);
        if ( v38 == sub_948040 )
        {
          if ( *v67 > 0x15u || *v69 > 0x15u || *v37 > 0x15u )
          {
LABEL_31:
            v79 = 257;
            v41 = sub_BD2C40(72, 3u);
            v42 = v41;
            if ( v41 )
            {
              v43 = (__int64)v69;
              v70 = v41;
              sub_B4DFA0((__int64)v41, (__int64)v67, v43, (__int64)v37, (__int64)v78, (__int64)v41, 0, 0);
              v42 = v70;
            }
            v71 = (__int64)v42;
            (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
              *(_QWORD *)(a3 + 88),
              v42,
              v76,
              *(_QWORD *)(a3 + 56),
              *(_QWORD *)(a3 + 64));
            v39 = v71;
            v44 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
            if ( *(_QWORD *)a3 != v44 )
            {
              v72 = v14;
              v45 = v39;
              v46 = *(unsigned int **)a3;
              do
              {
                v47 = *((_QWORD *)v46 + 1);
                v48 = *v46;
                v46 += 4;
                sub_B99FD0(v45, v48, v47);
              }
              while ( (unsigned int *)v44 != v46 );
              v14 = v72;
              v39 = v45;
            }
            goto LABEL_28;
          }
          v39 = sub_AD5A90((__int64)v67, v69, v37, 0);
        }
        else
        {
          v39 = v38(v36, v67, v69, v37);
        }
        if ( !v39 )
          goto LABEL_31;
LABEL_28:
        v67 = (_BYTE *)v39;
        if ( ++v14 == v65 )
          return v12;
      }
      else if ( ++v14 == v65 )
      {
        return v12;
      }
    }
    v21 = v20(v18, (_BYTE *)a2, v19);
LABEL_11:
    if ( v21 )
      goto LABEL_12;
    goto LABEL_37;
  }
  return (_BYTE *)v11;
}
