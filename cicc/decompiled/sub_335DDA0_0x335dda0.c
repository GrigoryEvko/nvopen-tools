// Function: sub_335DDA0
// Address: 0x335dda0
//
_BYTE *__fastcall sub_335DDA0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 (*v5)(void); // rax
  _BYTE *result; // rax
  __int64 v8; // r15
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned int v13; // r10d
  _DWORD *v14; // rsi
  __int64 v15; // r13
  __int64 v16; // rcx
  unsigned __int64 v17; // rax
  int v18; // eax
  unsigned int *v19; // rax
  int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned int i; // ebx
  __int64 v24; // r9
  __int64 v25; // r12
  __int16 v26; // r11
  __int64 v27; // r9
  __int64 v28; // rdi
  __int64 (*v29)(); // rax
  __int64 v30; // rdi
  int v31; // eax
  unsigned __int64 v32; // rax
  int v33; // edx
  void (*v34)(); // rax
  unsigned __int16 v35; // ax
  bool v36; // zf
  char v37; // al
  int v38; // eax
  __int64 *v39; // rax
  char v40; // al
  unsigned __int16 *v41; // rcx
  unsigned int v42; // edi
  __int64 v43; // rsi
  unsigned int v44; // edx
  int *v45; // [rsp-8h] [rbp-A8h]
  unsigned int v46; // [rsp+8h] [rbp-98h]
  __int16 v47; // [rsp+Ch] [rbp-94h]
  unsigned int v48; // [rsp+Ch] [rbp-94h]
  __int16 v49; // [rsp+Ch] [rbp-94h]
  __int64 v50; // [rsp+10h] [rbp-90h]
  __int16 v51; // [rsp+10h] [rbp-90h]
  __int64 v52; // [rsp+10h] [rbp-90h]
  _BYTE *v53; // [rsp+18h] [rbp-88h]
  unsigned int v54; // [rsp+20h] [rbp-80h]
  __int64 v55; // [rsp+20h] [rbp-80h]
  __int64 v56; // [rsp+28h] [rbp-78h]
  int v57; // [rsp+30h] [rbp-70h]
  char v58; // [rsp+3Bh] [rbp-65h]
  unsigned int v59; // [rsp+3Ch] [rbp-64h]
  __int64 v60; // [rsp+40h] [rbp-60h]
  _BYTE *v61; // [rsp+48h] [rbp-58h]
  int v62; // [rsp+58h] [rbp-48h]
  int v63; // [rsp+5Ch] [rbp-44h] BYREF
  __int64 v64; // [rsp+60h] [rbp-40h] BYREF
  int v65; // [rsp+68h] [rbp-38h]
  _BOOL4 v66; // [rsp+6Ch] [rbp-34h]

  v58 = 0;
  v56 = *(_QWORD *)(a1[4] + 16LL);
  v5 = *(__int64 (**)(void))(*a1 + 112LL);
  if ( v5 != sub_334CAA0 )
    v58 = v5();
  result = (_BYTE *)a1[6];
  v53 = (_BYTE *)a1[7];
  if ( result != v53 )
  {
    v61 = (_BYTE *)a1[6];
    while ( 1 )
    {
      v8 = *(_QWORD *)v61;
      v9 = *(_DWORD *)(*(_QWORD *)v61 + 24LL);
      if ( v9 < 0 )
      {
        v41 = (unsigned __int16 *)(*(_QWORD *)(a1[2] + 8LL) - 40LL * (unsigned int)~v9);
        v42 = v41[1];
        if ( v41[1] )
        {
          v43 = 0;
          v44 = 0;
          while ( 1 )
          {
            if ( v42 > v44 )
            {
              a5 = 5LL * *v41 + 5;
              if ( (v41[4 * a5 + 2 + 3 * v41[8] + v43] & 1) != 0 )
                break;
            }
            ++v44;
            v43 += 3;
            if ( v44 == v42 )
              goto LABEL_59;
          }
          v61[248] |= 8u;
        }
LABEL_59:
        if ( (*((_BYTE *)v41 + 27) & 2) != 0 )
          v61[248] |= 0x10u;
        goto LABEL_15;
      }
LABEL_6:
      v10 = *(unsigned int *)(v8 + 64);
      if ( (_DWORD)v10 )
        break;
LABEL_23:
      v61 += 256;
      result = v61;
      if ( v53 == v61 )
        return result;
    }
LABEL_7:
    v60 = v10;
    v11 = 0;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v8 + 40);
      v13 = v11;
      v14 = (_DWORD *)(v12 + 40 * v11);
      v15 = *(_QWORD *)v14;
      v16 = *(unsigned int *)(*(_QWORD *)v14 + 24LL);
      v17 = (0x3FF8000FFE42uLL >> v16) & 1;
      if ( (unsigned int)v16 >= 0x2E )
        LOBYTE(v17) = 0;
      if ( (_DWORD)v16 == 324 )
        goto LABEL_12;
      if ( (_BYTE)v17 )
        goto LABEL_12;
      v25 = a1[6] + ((__int64)*(int *)(v15 + 36) << 8);
      if ( (_BYTE *)v25 == v61 )
        goto LABEL_12;
      v59 = v14[2];
      v26 = *(_WORD *)(*(_QWORD *)(v15 + 48) + 16LL * v59);
      v62 = 0;
      v63 = 1;
      if ( v11 != 2 || *(_DWORD *)(v8 + 24) != 49 )
        goto LABEL_43;
      v27 = a1[2];
      a5 = a1[3];
      v28 = *(_QWORD *)(a1[74] + 16LL);
      v57 = *(_DWORD *)(*(_QWORD *)(v12 + 40) + 96LL);
      v29 = *(__int64 (**)())(*(_QWORD *)v28 + 2280LL);
      if ( v29 == sub_302E1F0 )
      {
        if ( v57 < 0 )
          goto LABEL_43;
      }
      else
      {
        v55 = a1[3];
        v45 = &v63;
        v47 = v26;
        v50 = a1[2];
        v37 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v29)(v28, v15, v8, 2);
        a5 = v55;
        v27 = v50;
        v26 = v47;
        v13 = 2;
        if ( v37 || v57 < 0 )
        {
LABEL_33:
          v31 = v63;
          goto LABEL_34;
        }
        v12 = *(_QWORD *)(v8 + 40);
        v16 = *(unsigned int *)(v15 + 24);
      }
      v54 = *(_DWORD *)(v12 + 88);
      if ( (_DWORD)v16 == 50 )
      {
        if ( v57 != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v15 + 40) + 40LL) + 96LL) )
          goto LABEL_33;
        v38 = v57;
        v62 = v57;
      }
      else
      {
        if ( (int)v16 >= 0 )
          goto LABEL_33;
        v16 = (unsigned int)~(_DWORD)v16;
        v30 = *(_QWORD *)(v27 + 8) - 40 * v16;
        if ( v54 < *(unsigned __int8 *)(v30 + 4) )
          goto LABEL_33;
        v46 = v13;
        v49 = v26;
        v52 = a5;
        v40 = sub_3148160((unsigned __int16 *)v30, v57, 0);
        a5 = v52;
        v26 = v49;
        v13 = v46;
        if ( !v40 )
          goto LABEL_33;
        v38 = v57;
        v62 = v57;
      }
      if ( !v38 )
        goto LABEL_33;
      v48 = v13;
      v51 = v26;
      v39 = sub_2FF6500(a5, v57, *(_WORD *)(*(_QWORD *)(v15 + 48) + 16LL * v54));
      v13 = v48;
      v26 = v51;
      v31 = *(char *)(*v39 + 28);
      v63 = v31;
LABEL_34:
      if ( v31 < 0 )
      {
        v32 = v25 & 0xFFFFFFFFFFFFFFF9LL;
        if ( v26 != 1 )
          goto LABEL_36;
        goto LABEL_44;
      }
LABEL_43:
      v62 = 0;
      v32 = v25 & 0xFFFFFFFFFFFFFFF9LL;
      if ( v26 != 1 )
      {
LABEL_36:
        v64 = v32;
        v33 = *(unsigned __int16 *)(v25 + 252);
        v65 = v62;
        v66 = v33;
        if ( !v58 )
        {
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD, __int64 *))(*a1 + 80LL))(a1, v15, v8, v13, &v64);
          v34 = *(void (**)())(*(_QWORD *)v56 + 344LL);
          if ( v34 != nullsub_1667 )
          {
            ((void (__fastcall *)(__int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))v34)(
              v56,
              v25,
              v59,
              v61,
              (unsigned int)v11,
              &v64,
              0);
            v16 = (__int64)v45;
          }
        }
        goto LABEL_39;
      }
LABEL_44:
      v36 = *(_DWORD *)(v15 + 24) == 2;
      v65 = 0;
      v64 = v32 | 6;
      v66 = !v36;
LABEL_39:
      if ( !(unsigned __int8)sub_2F8F1B0((__int64)v61, (__int64)&v64, 1u, v16, a5, (unsigned __int64)&v64)
        && (v64 & 6) == 0 )
      {
        v35 = *(_WORD *)(v25 + 250);
        if ( v35 > 1u )
          *(_WORD *)(v25 + 250) = v35 - 1;
      }
LABEL_12:
      if ( ++v11 == v60 )
      {
        v18 = *(_DWORD *)(v8 + 64);
        if ( !v18 )
          goto LABEL_23;
        v19 = (unsigned int *)(*(_QWORD *)(v8 + 40) + 40LL * (unsigned int)(v18 - 1));
        v8 = *(_QWORD *)v19;
        if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v19 + 48LL) + 16LL * v19[2]) != 262 )
          goto LABEL_23;
LABEL_15:
        v20 = *(_DWORD *)(v8 + 24);
        if ( v20 < 0 && *(_BYTE *)(*(_QWORD *)(a1[2] + 8LL) - 40LL * (unsigned int)~v20 + 9) )
        {
          v61[248] |= 0x80u;
          for ( i = sub_3751FC0(v8); i; --i )
          {
            if ( (unsigned __int8)sub_33CF8A0(v8, i - 1, v21, v22, a5, v24) )
            {
              if ( i <= *(unsigned __int8 *)(*(_QWORD *)(a1[2] + 8LL) - 40LL * (unsigned int)~*(_DWORD *)(v8 + 24) + 4) )
                goto LABEL_6;
              v61[248] |= 0x40u;
              v10 = *(unsigned int *)(v8 + 64);
              if ( (_DWORD)v10 )
                goto LABEL_7;
              goto LABEL_23;
            }
          }
        }
        goto LABEL_6;
      }
    }
  }
  return result;
}
