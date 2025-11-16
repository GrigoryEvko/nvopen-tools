// Function: sub_33520E0
// Address: 0x33520e0
//
void __fastcall sub_33520E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6)
{
  __int64 v6; // r14
  int v7; // eax
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  unsigned int *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rax
  int v17; // r14d
  __int64 v18; // r13
  _QWORD *v19; // rbx
  __int64 v20; // r15
  __int64 v21; // r12
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 (__fastcall *v24)(__int64, unsigned __int16); // rcx
  __int64 v25; // rcx
  __int64 (__fastcall *v26)(__int64, unsigned __int16); // rax
  unsigned int v27; // r10d
  unsigned __int8 v28; // al
  __int64 v29; // rdi
  __int64 (__fastcall *v30)(__int64, unsigned __int16); // rax
  unsigned __int8 v31; // al
  int v32; // eax
  __int64 v33; // rcx
  int v34; // ebx
  __int64 v35; // rdx
  unsigned int i; // r13d
  __int64 v37; // r15
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 (__fastcall *v40)(__int64, unsigned __int16); // rdx
  __int64 *v41; // rdx
  __int64 v42; // rdx
  __int64 (__fastcall *v43)(__int64, unsigned __int16); // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 (__fastcall *v49)(__int64, unsigned __int16); // rdx
  __int64 v50; // rdx
  __int64 (__fastcall *v51)(__int64, unsigned __int16); // rax
  unsigned __int16 v52; // r12
  int v53; // edx
  __int64 v54; // rdx
  _QWORD *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int8 v59; // al
  __int64 v60; // rax
  _QWORD *v61; // [rsp+8h] [rbp-68h]
  __int64 v62; // [rsp+10h] [rbp-60h]
  unsigned int v63; // [rsp+1Ch] [rbp-54h]
  __int64 v64; // [rsp+20h] [rbp-50h]
  _DWORD *v65; // [rsp+20h] [rbp-50h]
  __int64 v66; // [rsp+28h] [rbp-48h]
  _QWORD *v68; // [rsp+38h] [rbp-38h]
  unsigned int v69; // [rsp+38h] [rbp-38h]

  if ( !*(_BYTE *)(a1 + 44) )
    return;
  v6 = *a2;
  if ( !*a2 )
    return;
  v7 = *(_DWORD *)(v6 + 24);
  v8 = (_QWORD *)a1;
  if ( v7 < 0 )
  {
    v44 = (unsigned int)~v7;
    if ( (unsigned int)v44 <= 0x13 )
    {
      v45 = 530176;
      if ( _bittest64(&v45, v44) )
        return;
    }
  }
  else if ( v7 != 49 )
  {
    return;
  }
  v9 = (_QWORD *)a2[5];
  v68 = &v9[2 * *((unsigned int *)a2 + 12)];
  if ( v9 == v68 )
    goto LABEL_30;
  v66 = *a2;
  v10 = (_QWORD *)a1;
  do
  {
    while ( 1 )
    {
      if ( (*v9 & 6) != 0 )
        goto LABEL_28;
      v11 = (unsigned int *)(*v9 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = v11[32];
      if ( v11[55] != (_DWORD)v12 )
        goto LABEL_28;
      v13 = *(_QWORD *)v11;
      v14 = *(_DWORD *)(*(_QWORD *)v11 + 24LL);
      if ( v14 >= 0 )
        break;
      v15 = (unsigned int)~v14;
      if ( v14 == -11 )
        goto LABEL_28;
      if ( (unsigned int)(-v14 - 9) <= 1 || (_DWORD)v15 == 12 )
        goto LABEL_47;
      if ( (_DWORD)v15 != 19 )
      {
        v16 = *(_QWORD *)(v10[8] + 8LL) - 40 * v15;
        v17 = *(unsigned __int8 *)(v16 + 4);
        if ( !*(_BYTE *)(v16 + 4) )
          goto LABEL_28;
        v61 = v9;
        v18 = 0;
        v19 = v10;
        v20 = v13;
        while ( 1 )
        {
          v21 = *(unsigned __int16 *)(*(_QWORD *)(v20 + 48) + 16 * v18);
          if ( !(unsigned __int8)sub_33CF8A0(v20, (unsigned int)v18, v12, a4, a5, a6) )
            goto LABEL_17;
          v22 = v19[10];
          v23 = *(_QWORD *)v22;
          v24 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v22 + 568LL);
          if ( v24 == sub_2FE3130 )
          {
            v25 = *(_QWORD *)(v22 + 8LL * (unsigned __int16)v21 + 3400);
          }
          else
          {
            v57 = v24(v22, v21);
            v22 = v19[10];
            v25 = v57;
            v23 = *(_QWORD *)v22;
          }
          v26 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(v23 + 576);
          a4 = *(unsigned __int16 *)(*(_QWORD *)v25 + 24LL);
          v27 = *(_DWORD *)(v19[15] + 4 * a4);
          a6 = (_DWORD *)(4 * a4);
          if ( v26 == sub_2FE3140 )
          {
            v28 = *(_BYTE *)(v22 + (unsigned __int16)v21 + 5592);
          }
          else
          {
            v62 = a4;
            v63 = *(_DWORD *)(v19[15] + 4 * a4);
            v64 = 4 * a4;
            v28 = v26(v22, v21);
            a4 = v62;
            v27 = v63;
            a6 = (_DWORD *)v64;
          }
          if ( v27 < v28 )
          {
            *(_DWORD *)(v19[15] + 4 * a4) = 0;
LABEL_17:
            if ( v17 == (_DWORD)++v18 )
              goto LABEL_27;
          }
          else
          {
            v29 = v19[10];
            v30 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v29 + 576LL);
            if ( v30 == sub_2FE3140 )
            {
              v31 = *(_BYTE *)(v29 + v21 + 5592);
            }
            else
            {
              v65 = a6;
              v31 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64))v30)(v29, (unsigned int)v21, v12, a4);
              a6 = v65;
            }
            a6 = (_DWORD *)((char *)a6 + v19[15]);
            ++v18;
            *a6 -= v31;
            if ( v17 == (_DWORD)v18 )
            {
LABEL_27:
              v10 = v19;
              v9 = v61;
              goto LABEL_28;
            }
          }
        }
      }
      v54 = *(_QWORD *)(**(_QWORD **)(v13 + 40) + 96LL);
      v55 = *(_QWORD **)(v54 + 24);
      if ( *(_DWORD *)(v54 + 32) > 0x40u )
        v55 = (_QWORD *)*v55;
      v56 = *(unsigned __int16 *)(**(_QWORD **)(*(_QWORD *)(v10[9] + 280LL) + 8LL * (unsigned int)v55) + 24LL);
      ++*(_DWORD *)(v10[15] + 4 * v56);
LABEL_28:
      v9 += 2;
      if ( v68 == v9 )
        goto LABEL_29;
    }
    if ( v14 != 50 )
      goto LABEL_28;
LABEL_47:
    v46 = v10[10];
    v47 = **(unsigned __int16 **)(v13 + 48);
    v48 = *(_QWORD *)v46;
    v49 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v46 + 568LL);
    if ( v49 == sub_2FE3130 )
    {
      v50 = *(_QWORD *)(v46 + 8LL * (unsigned __int16)v47 + 3400);
    }
    else
    {
      v58 = v49(v46, **(_WORD **)(v13 + 48));
      v46 = v10[10];
      v50 = v58;
      v48 = *(_QWORD *)v46;
    }
    v51 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(v48 + 576);
    v52 = *(_WORD *)(*(_QWORD *)v50 + 24LL);
    if ( v51 == sub_2FE3140 )
      v53 = *(unsigned __int8 *)(v46 + v47 + 5592);
    else
      v53 = (unsigned __int8)v51(v46, v47);
    v9 += 2;
    *(_DWORD *)(v10[15] + 4LL * v52) += v53;
  }
  while ( v68 != v9 );
LABEL_29:
  v6 = v66;
  v8 = v10;
LABEL_30:
  if ( *((_DWORD *)a2 + 53) )
  {
    v32 = *(_DWORD *)(v6 + 24);
    if ( v32 < 0 )
    {
      v33 = v8[8];
      v34 = *(_DWORD *)(v6 + 68);
      v35 = 40LL * (unsigned int)~v32;
      for ( i = *(unsigned __int8 *)(*(_QWORD *)(v33 + 8) - v35 + 4); v34 != i; ++i )
      {
        v37 = *(unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * i);
        if ( (_WORD)v37 != 1 && (_WORD)v37 != 262 && (unsigned __int8)sub_33CF8A0(v6, i, v35, v33, a5, a6) )
        {
          v38 = v8[10];
          v39 = *(_QWORD *)v38;
          v40 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v38 + 568LL);
          if ( v40 == sub_2FE3130 )
          {
            v41 = *(__int64 **)(v38 + 8LL * (unsigned __int16)v37 + 3400);
          }
          else
          {
            v60 = v40(v38, v37);
            v38 = v8[10];
            v41 = (__int64 *)v60;
            v39 = *(_QWORD *)v38;
          }
          v42 = *v41;
          v43 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(v39 + 576);
          v33 = *(unsigned __int16 *)(v42 + 24);
          if ( v43 == sub_2FE3140 )
          {
            v35 = *(unsigned __int8 *)(v38 + v37 + 5592);
          }
          else
          {
            v69 = *(unsigned __int16 *)(v42 + 24);
            v59 = v43(v38, v37);
            v33 = v69;
            v35 = v59;
          }
          *(_DWORD *)(v8[15] + 4LL * (unsigned __int16)v33) += v35;
        }
      }
    }
  }
}
