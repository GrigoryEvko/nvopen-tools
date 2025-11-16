// Function: sub_1ECBFD0
// Address: 0x1ecbfd0
//
void __fastcall sub_1ECBFD0(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  int v14; // r9d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 (*v19)(); // rsi
  __int64 v20; // rcx
  unsigned int v21; // edx
  int v22; // r8d
  int v23; // r9d
  void (*v24)(void); // rax
  int *v25; // r15
  __int64 v26; // rbx
  int v27; // r11d
  unsigned __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // r10
  __int64 v31; // r12
  __int64 v32; // r14
  _QWORD *v33; // r12
  unsigned int v34; // ecx
  unsigned int v35; // edx
  _QWORD *v36; // rax
  _BOOL4 v37; // r10d
  __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // rdi
  _QWORD *v44; // rsi
  _QWORD *v45; // rdx
  __int64 v46; // [rsp+0h] [rbp-140h]
  const void *v47; // [rsp+8h] [rbp-138h]
  __int64 v48; // [rsp+10h] [rbp-130h]
  _QWORD *v49; // [rsp+18h] [rbp-128h]
  _BOOL4 v51; // [rsp+20h] [rbp-120h]
  __int64 v52; // [rsp+20h] [rbp-120h]
  unsigned int v53; // [rsp+20h] [rbp-120h]
  int v54; // [rsp+20h] [rbp-120h]
  unsigned int v55; // [rsp+28h] [rbp-118h]
  int v56; // [rsp+2Ch] [rbp-114h] BYREF
  _QWORD v57[2]; // [rsp+30h] [rbp-110h] BYREF
  __int64 v58; // [rsp+40h] [rbp-100h]
  __int64 v59; // [rsp+48h] [rbp-F8h]
  __int64 v60; // [rsp+50h] [rbp-F0h]
  __int64 v61; // [rsp+58h] [rbp-E8h]
  __int64 v62; // [rsp+60h] [rbp-E0h]
  __int64 v63; // [rsp+68h] [rbp-D8h]
  unsigned int v64; // [rsp+70h] [rbp-D0h]
  char v65; // [rsp+74h] [rbp-CCh]
  _QWORD *v66; // [rsp+78h] [rbp-C8h]
  __int64 v67; // [rsp+80h] [rbp-C0h]
  _BYTE *v68; // [rsp+88h] [rbp-B8h]
  _BYTE *v69; // [rsp+90h] [rbp-B0h]
  __int64 v70; // [rsp+98h] [rbp-A8h]
  int v71; // [rsp+A0h] [rbp-A0h]
  _BYTE v72[32]; // [rsp+A8h] [rbp-98h] BYREF
  __int64 v73; // [rsp+C8h] [rbp-78h]
  _BYTE *v74; // [rsp+D0h] [rbp-70h]
  _BYTE *v75; // [rsp+D8h] [rbp-68h]
  __int64 v76; // [rsp+E0h] [rbp-60h]
  int v77; // [rsp+E8h] [rbp-58h]
  _BYTE v78[80]; // [rsp+F0h] [rbp-50h] BYREF

  v56 = a2;
  sub_1ECB700(a1 + 30, (unsigned int *)&v56);
  v15 = sub_1E86160(a5, v56, v11, v12, v13, v14);
  v17 = *(_QWORD *)(a4 + 16);
  v58 = a3;
  v57[1] = v15;
  v18 = *(_QWORD *)(a4 + 40);
  v60 = a5;
  v61 = a6;
  v57[0] = &unk_4A00C10;
  v59 = v18;
  v19 = *(__int64 (**)())(*(_QWORD *)v17 + 40LL);
  v20 = 0;
  if ( v19 != sub_1D00B00 )
  {
    v20 = ((__int64 (__fastcall *)(__int64, __int64 (*)(), __int64, _QWORD))v19)(v17, v19, v16, 0);
    v18 = v59;
  }
  v21 = *(_DWORD *)(a3 + 8);
  v62 = v20;
  v63 = 0;
  v64 = v21;
  v68 = v72;
  v69 = v72;
  v65 = 0;
  v66 = a1 + 42;
  v67 = 0;
  v70 = 4;
  v71 = 0;
  v73 = 0;
  v74 = v78;
  v75 = v78;
  v76 = 4;
  v77 = 0;
  *(_QWORD *)(v18 + 8) = v57;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a7 + 24LL))(a7);
  v24 = *(void (**)(void))(**(_QWORD **)(a4 + 16) + 112LL);
  if ( (char *)v24 != (char *)sub_1D00B10 )
    v24();
  v47 = (const void *)(a5 + 416);
  v25 = (int *)(*(_QWORD *)v58 + 4LL * v64);
  v48 = *(_QWORD *)v58 + 4LL * *(unsigned int *)(v58 + 8);
  if ( v25 != (int *)v48 )
  {
    v49 = a1 + 31;
    v26 = a5;
    while ( 1 )
    {
      v27 = *v25;
      v28 = *(unsigned int *)(v26 + 408);
      v29 = *v25 & 0x7FFFFFFF;
      v30 = v29;
      v31 = 8LL * v29;
      if ( v29 < (unsigned int)v28 )
      {
        v32 = *(_QWORD *)(*(_QWORD *)(v26 + 400) + 8LL * v29);
        if ( v32 )
        {
          v33 = (_QWORD *)a1[32];
          if ( !v33 )
            goto LABEL_28;
          goto LABEL_10;
        }
      }
      v39 = v29 + 1;
      if ( (unsigned int)v28 >= v39 )
        goto LABEL_26;
      v42 = v39;
      if ( v39 < v28 )
        break;
      if ( v39 <= v28 )
        goto LABEL_26;
      if ( v39 > (unsigned __int64)*(unsigned int *)(v26 + 412) )
      {
        v46 = v30;
        v55 = v39;
        v54 = *v25;
        sub_16CD150(v26 + 400, v47, v39, 8, v22, v23);
        v28 = *(unsigned int *)(v26 + 408);
        v30 = v46;
        v39 = v55;
        v27 = v54;
      }
      v40 = *(_QWORD *)(v26 + 400);
      v43 = *(_QWORD *)(v26 + 416);
      v44 = (_QWORD *)(v40 + 8 * v42);
      v45 = (_QWORD *)(v40 + 8 * v28);
      if ( v44 != v45 )
      {
        do
          *v45++ = v43;
        while ( v44 != v45 );
        v40 = *(_QWORD *)(v26 + 400);
      }
      *(_DWORD *)(v26 + 408) = v39;
LABEL_27:
      v52 = v30;
      *(_QWORD *)(v40 + v31) = sub_1DBA290(v27);
      v32 = *(_QWORD *)(*(_QWORD *)(v26 + 400) + 8 * v52);
      sub_1DBB110((_QWORD *)v26, v32);
      v33 = (_QWORD *)a1[32];
      if ( !v33 )
      {
LABEL_28:
        v33 = a1 + 31;
        if ( v49 != (_QWORD *)a1[33] )
        {
          v34 = *(_DWORD *)(v32 + 112);
          goto LABEL_31;
        }
        v33 = a1 + 31;
        v37 = 1;
        goto LABEL_18;
      }
LABEL_10:
      v34 = *(_DWORD *)(v32 + 112);
      while ( 1 )
      {
        v35 = *((_DWORD *)v33 + 8);
        v36 = (_QWORD *)v33[3];
        if ( v34 < v35 )
          v36 = (_QWORD *)v33[2];
        if ( !v36 )
          break;
        v33 = v36;
      }
      if ( v34 < v35 )
      {
        if ( v33 == (_QWORD *)a1[33] )
        {
LABEL_17:
          v37 = 1;
          if ( v49 != v33 )
            goto LABEL_33;
        }
        else
        {
LABEL_31:
          v53 = v34;
          v41 = sub_220EF80(v33);
          v34 = v53;
          if ( v53 <= *(_DWORD *)(v41 + 32) )
            goto LABEL_19;
          v37 = 1;
          if ( v49 != v33 )
LABEL_33:
            v37 = v34 < *((_DWORD *)v33 + 8);
        }
LABEL_18:
        v51 = v37;
        v38 = sub_22077B0(40);
        *(_DWORD *)(v38 + 32) = *(_DWORD *)(v32 + 112);
        sub_220F040(v51, v38, v33, v49);
        ++a1[35];
        goto LABEL_19;
      }
      if ( v34 > v35 )
        goto LABEL_17;
LABEL_19:
      if ( (int *)v48 == ++v25 )
        goto LABEL_20;
    }
    *(_DWORD *)(v26 + 408) = v39;
LABEL_26:
    v40 = *(_QWORD *)(v26 + 400);
    goto LABEL_27;
  }
LABEL_20:
  *(_QWORD *)(v59 + 8) = 0;
  if ( v75 != v74 )
    _libc_free((unsigned __int64)v75);
  if ( v69 != v68 )
    _libc_free((unsigned __int64)v69);
}
