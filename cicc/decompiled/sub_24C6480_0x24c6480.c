// Function: sub_24C6480
// Address: 0x24c6480
//
__int64 __fastcall sub_24C6480(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  _BYTE *v17; // r13
  __int64 (__fastcall *v18)(__int64, __int64, __int64); // rax
  unsigned int *v19; // r15
  unsigned int *v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // r9
  __int64 v26; // r14
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rbx
  __int64 v31; // r15
  __int64 v32; // r13
  __int64 **v33; // rax
  __int64 v34; // rax
  _BYTE *v36; // r13
  __int64 (__fastcall *v37)(__int64, __int64, __int64); // rax
  unsigned int *v38; // r14
  unsigned int *v39; // r15
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 **v45; // r14
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  unsigned int *v48; // r15
  unsigned int *v49; // r14
  __int64 v50; // rdx
  __int64 v51; // rsi
  unsigned int *v52; // rbx
  unsigned int *v53; // r14
  __int64 v54; // rdx
  unsigned int v55; // esi
  __int64 v56; // [rsp-10h] [rbp-280h]
  __int64 v57; // [rsp+8h] [rbp-268h]
  int v61[8]; // [rsp+40h] [rbp-230h] BYREF
  __int16 v62; // [rsp+60h] [rbp-210h]
  _BYTE v63[32]; // [rsp+70h] [rbp-200h] BYREF
  __int16 v64; // [rsp+90h] [rbp-1E0h]
  unsigned int *v65; // [rsp+A0h] [rbp-1D0h] BYREF
  int v66; // [rsp+A8h] [rbp-1C8h]
  char v67; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v68; // [rsp+D8h] [rbp-198h]
  __int64 v69; // [rsp+E0h] [rbp-190h]
  __int64 v70; // [rsp+F0h] [rbp-180h]
  __int64 v71; // [rsp+F8h] [rbp-178h]
  void *v72; // [rsp+120h] [rbp-150h]
  __int64 *v73; // [rsp+130h] [rbp-140h] BYREF
  __int64 v74; // [rsp+138h] [rbp-138h]
  _BYTE v75[304]; // [rsp+140h] [rbp-130h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v73 = (__int64 *)v75;
  v74 = 0x2000000000LL;
  if ( v6 )
    v6 -= 24;
  v7 = sub_AA5190(v6);
  if ( v7 )
    v7 -= 24;
  v8 = 0;
  sub_23D0AB0((__int64)&v65, v7, 0, 0, 0);
  if ( a4 )
  {
    do
    {
      v10 = 257;
      v62 = 257;
      v11 = *(_QWORD *)(a1 + 456);
      v12 = *(_QWORD *)(a2 + 80);
      v13 = *(_QWORD *)(a3 + 8 * v8);
      if ( v12 )
        v12 -= 24;
      if ( v13 == v12 )
      {
        v36 = (_BYTE *)a2;
        if ( *(_QWORD *)(a2 + 8) != v11 )
        {
          if ( *(_BYTE *)a2 > 0x15u )
          {
            v51 = *(_QWORD *)(a1 + 456);
            v64 = 257;
            v36 = (_BYTE *)sub_B52210(a2, v51, (__int64)v63, 0, 0);
            (*(void (__fastcall **)(__int64, _BYTE *, int *, __int64, __int64))(*(_QWORD *)v71 + 16LL))(
              v71,
              v36,
              v61,
              v68,
              v69);
            if ( v65 != &v65[4 * v66] )
            {
              v57 = v8;
              v52 = v65;
              v53 = &v65[4 * v66];
              do
              {
                v54 = *((_QWORD *)v52 + 1);
                v55 = *v52;
                v52 += 4;
                sub_B99FD0((__int64)v36, v55, v54);
              }
              while ( v53 != v52 );
              v8 = v57;
            }
          }
          else
          {
            v37 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v70 + 136LL);
            if ( v37 == sub_928970 )
              v36 = (_BYTE *)sub_ADAFB0(a2, *(_QWORD *)(a1 + 456));
            else
              v36 = (_BYTE *)v37(v70, a2, *(_QWORD *)(a1 + 456));
            if ( *v36 > 0x1Cu )
            {
              (*(void (__fastcall **)(__int64, _BYTE *, int *, __int64, __int64))(*(_QWORD *)v71 + 16LL))(
                v71,
                v36,
                v61,
                v68,
                v69);
              v38 = &v65[4 * v66];
              if ( v65 != v38 )
              {
                v39 = v65;
                do
                {
                  v40 = *((_QWORD *)v39 + 1);
                  v41 = *v39;
                  v39 += 4;
                  sub_B99FD0((__int64)v36, v41, v40);
                }
                while ( v38 != v39 );
              }
            }
          }
        }
        v42 = (unsigned int)v74;
        v43 = (unsigned int)v74 + 1LL;
        if ( v43 > HIDWORD(v74) )
        {
          sub_C8D5F0((__int64)&v73, v75, v43, 8u, v9, v10);
          v42 = (unsigned int)v74;
        }
        v73[v42] = (__int64)v36;
        v44 = *(_QWORD *)(a1 + 464);
        v64 = 257;
        v45 = *(__int64 ***)(a1 + 456);
        LODWORD(v74) = v74 + 1;
        v46 = sub_AD64C0(v44, 1, 0);
        v47 = sub_24C3260((__int64 *)&v65, 0x30u, v46, v45, (__int64)v63, 0, v61[0], 0);
        v7 = v56;
        v26 = v47;
        v27 = (unsigned int)v74;
        v28 = (unsigned int)v74 + 1LL;
        if ( v28 <= HIDWORD(v74) )
          goto LABEL_19;
      }
      else
      {
        v14 = sub_ACC4F0(v13);
        v17 = (_BYTE *)v14;
        if ( *(_QWORD *)(v14 + 8) != v11 )
        {
          if ( *(_BYTE *)v14 > 0x15u )
          {
            v64 = 257;
            v17 = (_BYTE *)sub_B52210(v14, v11, (__int64)v63, 0, 0);
            v7 = (__int64)v17;
            (*(void (__fastcall **)(__int64, _BYTE *, int *, __int64, __int64))(*(_QWORD *)v71 + 16LL))(
              v71,
              v17,
              v61,
              v68,
              v69);
            v48 = v65;
            v49 = &v65[4 * v66];
            if ( v65 != v49 )
            {
              do
              {
                v50 = *((_QWORD *)v48 + 1);
                v7 = *v48;
                v48 += 4;
                sub_B99FD0((__int64)v17, v7, v50);
              }
              while ( v49 != v48 );
            }
          }
          else
          {
            v18 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v70 + 136LL);
            if ( v18 == sub_928970 )
            {
              v7 = v11;
              v17 = (_BYTE *)sub_ADAFB0((unsigned __int64)v17, v11);
            }
            else
            {
              v7 = (__int64)v17;
              v17 = (_BYTE *)v18(v70, (__int64)v17, v11);
            }
            if ( *v17 > 0x1Cu )
            {
              v7 = (__int64)v17;
              (*(void (__fastcall **)(__int64, _BYTE *, int *, __int64, __int64))(*(_QWORD *)v71 + 16LL))(
                v71,
                v17,
                v61,
                v68,
                v69);
              v19 = v65;
              v20 = &v65[4 * v66];
              if ( v65 != v20 )
              {
                do
                {
                  v21 = *((_QWORD *)v19 + 1);
                  v7 = *v19;
                  v19 += 4;
                  sub_B99FD0((__int64)v17, v7, v21);
                }
                while ( v20 != v19 );
              }
            }
          }
        }
        v22 = (unsigned int)v74;
        v23 = (unsigned int)v74 + 1LL;
        if ( v23 > HIDWORD(v74) )
        {
          v7 = (__int64)v75;
          sub_C8D5F0((__int64)&v73, v75, v23, 8u, v15, v16);
          v22 = (unsigned int)v74;
        }
        v73[v22] = (__int64)v17;
        v24 = *(_QWORD *)(a1 + 456);
        LODWORD(v74) = v74 + 1;
        v26 = sub_AD6530(v24, v7);
        v27 = (unsigned int)v74;
        v28 = (unsigned int)v74 + 1LL;
        if ( v28 <= HIDWORD(v74) )
          goto LABEL_19;
      }
      v7 = (__int64)v75;
      sub_C8D5F0((__int64)&v73, v75, v28, 8u, v9, v25);
      v27 = (unsigned int)v74;
LABEL_19:
      ++v8;
      v73[v27] = v26;
      LODWORD(v74) = v74 + 1;
    }
    while ( v8 != a4 );
  }
  v29 = sub_24C54D0(a1, 2 * a4, a2, *(__int64 **)(a1 + 456), "sancov_pcs");
  v30 = v73;
  v31 = (unsigned int)v74;
  v32 = v29;
  v33 = (__int64 **)sub_BCD420(*(__int64 **)(a1 + 456), 2 * a4);
  v34 = sub_AD1300(v33, v30, v31);
  sub_B30160(v32, v34);
  *(_BYTE *)(v32 + 80) |= 1u;
  nullsub_61();
  v72 = &unk_49DA100;
  nullsub_63();
  if ( v65 != (unsigned int *)&v67 )
    _libc_free((unsigned __int64)v65);
  if ( v73 != (__int64 *)v75 )
    _libc_free((unsigned __int64)v73);
  return v32;
}
