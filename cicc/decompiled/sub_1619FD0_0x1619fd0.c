// Function: sub_1619FD0
// Address: 0x1619fd0
//
__int64 __fastcall sub_1619FD0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 i; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  const char *v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rbx
  __int64 v22; // r12
  const char *v23; // rax
  size_t v24; // rdx
  _QWORD *v25; // rax
  unsigned __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // rdi
  unsigned __int8 v33; // r13
  __int64 (__fastcall *v34)(__int64, __int64); // rax
  __int64 *v35; // r10
  char v36; // r12
  __int64 *v37; // r13
  __int64 v38; // rbx
  _QWORD *v39; // r14
  _QWORD *v40; // rax
  char v41; // r11
  _BOOL4 v42; // r11d
  __int64 v43; // rax
  __int64 v44; // rdi
  const char *v45; // rax
  size_t v46; // rdx
  __int64 v48; // rax
  const char *v49; // rax
  size_t v50; // rdx
  __int64 v51; // [rsp+0h] [rbp-D0h]
  __int64 v52; // [rsp+8h] [rbp-C8h]
  __int64 v53; // [rsp+10h] [rbp-C0h]
  __int64 *v54; // [rsp+18h] [rbp-B8h]
  unsigned int v55; // [rsp+38h] [rbp-98h]
  char v56; // [rsp+3Fh] [rbp-91h]
  __int64 v57; // [rsp+40h] [rbp-90h]
  __int64 v58; // [rsp+40h] [rbp-90h]
  unsigned __int8 v59; // [rsp+48h] [rbp-88h]
  __int64 *v60; // [rsp+48h] [rbp-88h]
  _QWORD *v61; // [rsp+50h] [rbp-80h]
  _BOOL4 v62; // [rsp+58h] [rbp-78h]
  unsigned int v63; // [rsp+5Ch] [rbp-74h]
  __int64 v64; // [rsp+68h] [rbp-68h] BYREF
  _QWORD v65[12]; // [rsp+70h] [rbp-60h] BYREF

  v3 = a1;
  v59 = 0;
  if ( !sub_15E4F60(a2) )
  {
    v4 = *(_QWORD *)(a1 + 176);
    v5 = a1 + 328;
    v54 = *(__int64 **)(a2 + 40);
    v6 = *(_QWORD *)(v4 + 8);
    for ( i = *(_QWORD *)(v4 + 16); v6 != i; *(_QWORD *)(v5 - 8) = v8 + 224 )
    {
      v8 = *(_QWORD *)(i - 8);
      i -= 8;
      v5 += 8;
    }
    v9 = sub_1649960(a2);
    v11 = v10;
    if ( sub_16DA870(a2, v6, v10, v12, v13, v14) )
      sub_16DB3F0("OptFunction", 11, v9, v11);
    v15 = "size-info";
    v16 = sub_16033E0(*v54);
    v56 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v16 + 24LL))(v16, "size-info", 9);
    if ( *(_DWORD *)(v3 + 192) )
    {
      v59 = 0;
      v21 = v3 + 160;
      v55 = 0;
      v63 = 0;
      while ( 1 )
      {
        v22 = *(_QWORD *)(*(_QWORD *)(v3 + 184) + 8LL * v63);
        v23 = (const char *)sub_1649960(a2);
        sub_160F160(v21, v22, 0, 4, v23, v24);
        sub_1615D60(v21, (__int64 *)v22);
        sub_1614C80(v21, v22);
        sub_16C6860(v65);
        v65[2] = v22;
        v65[3] = a2;
        v65[0] = &unk_49ED7C0;
        v65[4] = 0;
        v25 = sub_1612E30((_QWORD *)v22);
        v61 = v25;
        if ( v25 )
          sub_16D7910(v25);
        v26 = v22;
        sub_1403F30(&v64, (_QWORD *)v22, *(_QWORD *)(v3 + 168));
        if ( v56 )
        {
          v26 = (unsigned __int64)v54;
          v55 = sub_160E760(v21, (__int64)v54);
        }
        if ( *(_BYTE *)(v22 + 152) )
        {
          v30 = *(_QWORD **)(v22 + 72);
          if ( !v30 )
            goto LABEL_20;
          v26 = v22 + 64;
          do
          {
            while ( 1 )
            {
              v28 = v30[2];
              v27 = v30[3];
              if ( v30[4] >= a2 )
                break;
              v30 = (_QWORD *)v30[3];
              if ( !v27 )
                goto LABEL_18;
            }
            v26 = (unsigned __int64)v30;
            v30 = (_QWORD *)v30[2];
          }
          while ( v28 );
LABEL_18:
          if ( v22 + 64 == v26 || *(_QWORD *)(v26 + 32) > a2 )
          {
LABEL_20:
            v31 = *(_QWORD **)(v22 + 120);
            v26 = *(_QWORD *)(a2 + 40);
            if ( !v31 )
            {
              v33 = 0;
              goto LABEL_46;
            }
            v32 = (_QWORD *)(v22 + 112);
            do
            {
              while ( 1 )
              {
                v28 = v31[2];
                v27 = v31[3];
                if ( v31[4] >= v26 )
                  break;
                v31 = (_QWORD *)v31[3];
                if ( !v27 )
                  goto LABEL_25;
              }
              v32 = v31;
              v31 = (_QWORD *)v31[2];
            }
            while ( v28 );
LABEL_25:
            v33 = 0;
            if ( v32 == (_QWORD *)(v22 + 112) || v32[4] > v26 )
              goto LABEL_46;
          }
        }
        v34 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v22 + 144LL);
        if ( v34 == sub_1619F90 )
        {
          v33 = 0;
          if ( sub_15E4F60(a2) )
            goto LABEL_46;
          v26 = a2;
          v33 = sub_1619BF0(v22 - 408, a2);
        }
        else
        {
          v26 = a2;
          v33 = v34(v22, a2);
        }
        if ( v33 )
          break;
LABEL_46:
        if ( v56 )
        {
          v26 = v22;
          sub_160FF80(v21, v22, (__int64)v54, v55);
        }
        v44 = v64;
        if ( v64 )
        {
          if ( v33 )
          {
            v26 = 2;
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v64 + 56LL))(v64, 2, v27, v28);
            v44 = v64;
          }
          (*(void (__fastcall **)(__int64, unsigned __int64, __int64, __int64))(*(_QWORD *)v44 + 48LL))(
            v44,
            v26,
            v27,
            v28);
        }
        if ( v61 )
          sub_16D7950(v61, v26, v27);
        v65[0] = &unk_49ED7C0;
        nullsub_616(v65, v26, v27, v28, v29);
        if ( v33 )
        {
          v49 = (const char *)sub_1649960(a2);
          sub_160F160(v21, v22, 1, 4, v49, v50);
        }
        sub_1615E90(v21, (__int64 *)v22);
        sub_1615FB0(v21, (__int64 *)v22);
        nullsub_568();
        sub_16145F0(v21, v22);
        sub_16176C0(v21, v22);
        v45 = (const char *)sub_1649960(a2);
        v15 = (const char *)v22;
        v16 = v21;
        sub_1615450(v21, v22, v45, v46, 4);
        if ( *(_DWORD *)(v3 + 192) <= ++v63 )
          goto LABEL_57;
      }
      v35 = *(__int64 **)(v22 + 32);
      v60 = *(__int64 **)(v22 + 40);
      if ( v35 == v60 )
        goto LABEL_45;
      v52 = v21;
      v51 = v3;
      v53 = v22;
      v36 = v33;
      v37 = v35;
      while ( 1 )
      {
        v38 = *v37;
        v39 = *(_QWORD **)(*v37 + 72);
        v28 = *v37 + 64;
        if ( !v39 )
          break;
        while ( 1 )
        {
          v26 = v39[4];
          v40 = (_QWORD *)v39[3];
          v41 = 0;
          if ( a2 < v26 )
          {
            v40 = (_QWORD *)v39[2];
            v41 = v36;
          }
          if ( !v40 )
            break;
          v39 = v40;
        }
        if ( v41 )
        {
          if ( v39 != *(_QWORD **)(v38 + 80) )
            goto LABEL_61;
LABEL_41:
          v42 = 1;
          if ( v39 == (_QWORD *)v28 )
            goto LABEL_42;
          goto LABEL_63;
        }
        if ( v26 < a2 )
          goto LABEL_41;
LABEL_43:
        if ( v60 == ++v37 )
        {
          v33 = v36;
          v21 = v52;
          v22 = v53;
          v3 = v51;
LABEL_45:
          v59 = v33;
          goto LABEL_46;
        }
      }
      v39 = *(_QWORD **)(v38 + 80);
      if ( (_QWORD *)v28 == v39 )
        goto LABEL_41;
      v39 = (_QWORD *)(*v37 + 64);
LABEL_61:
      v58 = *v37 + 64;
      v48 = sub_220EF80(v39);
      v28 = v58;
      if ( *(_QWORD *)(v48 + 32) >= a2 )
        goto LABEL_43;
      v42 = 1;
      if ( v39 == (_QWORD *)v58 )
      {
LABEL_42:
        v62 = v42;
        v57 = v28;
        v43 = sub_22077B0(40);
        *(_QWORD *)(v43 + 32) = a2;
        v26 = v43;
        sub_220F040(v62, v43, v39, v57);
        ++*(_QWORD *)(v38 + 96);
        goto LABEL_43;
      }
LABEL_63:
      v42 = a2 < v39[4];
      goto LABEL_42;
    }
    v59 = 0;
LABEL_57:
    if ( sub_16DA870(v16, v15, v17, v18, v19, v20) )
      sub_16DB5E0();
  }
  return v59;
}
