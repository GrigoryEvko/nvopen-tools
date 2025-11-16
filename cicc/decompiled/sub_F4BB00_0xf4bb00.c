// Function: sub_F4BB00
// Address: 0xf4bb00
//
void __fastcall sub_F4BB00(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        int a4,
        __int64 a5,
        _BYTE *a6,
        _BYTE *a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // r12
  unsigned __int8 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // r13
  int v17; // ebx
  unsigned int v18; // r12d
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 *v24; // rax
  _QWORD *v25; // rbx
  _QWORD *v26; // r13
  __int64 v27; // r13
  __int64 *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 *v32; // rax
  char v33; // dl
  __int64 v34; // [rsp+28h] [rbp-408h]
  _QWORD *v37; // [rsp+38h] [rbp-3F8h]
  _BYTE *v38; // [rsp+38h] [rbp-3F8h]
  char v39[8]; // [rsp+48h] [rbp-3E8h] BYREF
  __int64 v40; // [rsp+50h] [rbp-3E0h] BYREF
  __int64 *v41; // [rsp+58h] [rbp-3D8h]
  __int64 v42; // [rsp+60h] [rbp-3D0h]
  int v43; // [rsp+68h] [rbp-3C8h]
  char v44; // [rsp+6Ch] [rbp-3C4h]
  char v45; // [rsp+70h] [rbp-3C0h] BYREF
  char v46[8]; // [rsp+B0h] [rbp-380h] BYREF
  __int64 v47; // [rsp+B8h] [rbp-378h]
  char v48; // [rsp+CCh] [rbp-364h]
  _BYTE *v49; // [rsp+150h] [rbp-2E0h] BYREF
  __int64 v50; // [rsp+158h] [rbp-2D8h]
  _BYTE v51[64]; // [rsp+160h] [rbp-2D0h] BYREF
  _BYTE *v52; // [rsp+1A0h] [rbp-290h]
  __int64 v53; // [rsp+1A8h] [rbp-288h]
  _BYTE v54[64]; // [rsp+1B0h] [rbp-280h] BYREF
  _BYTE *v55; // [rsp+1F0h] [rbp-240h]
  __int64 v56; // [rsp+1F8h] [rbp-238h]
  _BYTE v57[64]; // [rsp+200h] [rbp-230h] BYREF
  _BYTE *v58; // [rsp+240h] [rbp-1F0h]
  __int64 v59; // [rsp+248h] [rbp-1E8h]
  _BYTE v60[64]; // [rsp+250h] [rbp-1E0h] BYREF
  _BYTE *v61; // [rsp+290h] [rbp-1A0h]
  __int64 v62; // [rsp+298h] [rbp-198h]
  _BYTE v63[64]; // [rsp+2A0h] [rbp-190h] BYREF
  __int64 v64; // [rsp+2E0h] [rbp-150h]
  char *v65; // [rsp+2E8h] [rbp-148h]
  __int64 v66; // [rsp+2F0h] [rbp-140h]
  int v67; // [rsp+2F8h] [rbp-138h]
  char v68; // [rsp+2FCh] [rbp-134h]
  char v69; // [rsp+300h] [rbp-130h] BYREF

  v11 = (__int64)a2;
  sub_B2B9F0(a1, *((_BYTE *)a2 + 128));
  sub_F4B000(a1, a2, a3, a4 > 0, a8, a9);
  if ( sub_B2FC80((__int64)a2) )
    return;
  v64 = 0;
  v49 = v51;
  v52 = v54;
  v55 = v57;
  v50 = 0x800000000LL;
  v53 = 0x800000000LL;
  v56 = 0x800000000LL;
  v58 = v60;
  v59 = 0x800000000LL;
  v61 = v63;
  v62 = 0x800000000LL;
  v65 = &v69;
  v66 = 32;
  v67 = 0;
  v68 = 1;
  v12 = (unsigned __int8 *)sub_F459D0((__int64)a2, a4, (__int64)&v49);
  sub_F45A20((__int64)v46, a4, (__int64)&v49, v12, v13, v14);
  sub_F45E60(a1, (__int64)a2, a3, 0, a8, a9, (__int64)v46);
  sub_F4B700(a1, (__int64)a2, a3, 0, a5, a6, a7, a8, a9, (__int64)v46);
  if ( a4 == 2 )
  {
    v15 = sub_BA8E40(*(_QWORD *)(a1 + 40), "llvm.dbg.cu", 0xBu);
    v44 = 1;
    v40 = 0;
    v16 = v15;
    v41 = (__int64 *)&v45;
    v42 = 8;
    v43 = 0;
    v17 = sub_B91A00(v15);
    if ( !v17 )
      goto LABEL_27;
    v37 = a2;
    v18 = 0;
    while ( 1 )
    {
      v21 = sub_B91A10(v16, v18);
      if ( !v44 )
        goto LABEL_39;
      v24 = v41;
      v19 = &v41[HIDWORD(v42)];
      if ( v41 != v19 )
      {
        while ( v21 != *v24 )
        {
          if ( v19 == ++v24 )
            goto LABEL_42;
        }
        goto LABEL_25;
      }
LABEL_42:
      if ( HIDWORD(v42) < (unsigned int)v42 )
      {
        ++HIDWORD(v42);
        *v19 = v21;
        ++v40;
      }
      else
      {
LABEL_39:
        sub_C8CC70((__int64)&v40, v21, (__int64)v19, v20, v22, v23);
      }
LABEL_25:
      if ( v17 == ++v18 )
      {
        v11 = (__int64)v37;
LABEL_27:
        a2 = &v49;
        sub_F45470(v11, (__int64)&v49);
        v25 = v49;
        v38 = &v49[8 * (unsigned int)v50];
        if ( v49 == v38 )
        {
LABEL_35:
          if ( !v44 )
            _libc_free(v41, a2);
          break;
        }
        v34 = v16;
        while ( 2 )
        {
          v26 = (_QWORD *)*v25;
          sub_FC75A0(v39, a3, 0, a8, a9, 0);
          a2 = v26;
          v27 = sub_FCD270(v39, v26);
          sub_FC7680(v39);
          if ( !v44 )
            goto LABEL_37;
          v32 = v41;
          v29 = HIDWORD(v42);
          v28 = &v41[HIDWORD(v42)];
          if ( v41 != v28 )
          {
            while ( v27 != *v32 )
            {
              if ( v28 == ++v32 )
                goto LABEL_40;
            }
            goto LABEL_34;
          }
LABEL_40:
          if ( HIDWORD(v42) < (unsigned int)v42 )
          {
            ++HIDWORD(v42);
            *v28 = v27;
            ++v40;
LABEL_38:
            a2 = (_QWORD *)v27;
            sub_B979A0(v34, v27);
          }
          else
          {
LABEL_37:
            a2 = (_QWORD *)v27;
            sub_C8CC70((__int64)&v40, v27, (__int64)v28, v29, v30, v31);
            if ( v33 )
              goto LABEL_38;
          }
LABEL_34:
          if ( v38 == (_BYTE *)++v25 )
            goto LABEL_35;
          continue;
        }
      }
    }
  }
  if ( !v48 )
    _libc_free(v47, a2);
  if ( !v68 )
    _libc_free(v65, a2);
  if ( v61 != v63 )
    _libc_free(v61, a2);
  if ( v58 != v60 )
    _libc_free(v58, a2);
  if ( v55 != v57 )
    _libc_free(v55, a2);
  if ( v52 != v54 )
    _libc_free(v52, a2);
  if ( v49 != v51 )
    _libc_free(v49, a2);
}
