// Function: sub_1361180
// Address: 0x1361180
//
__int64 __fastcall sub_1361180(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // r12d
  __int64 v10; // r11
  __int64 v12; // r10
  unsigned int v14; // eax
  __int64 v15; // r9
  __int64 v16; // rdi
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 v20; // r10
  __int64 v21; // r15
  char v22; // al
  __int64 v24; // r11
  __int64 v25; // r10
  char v26; // al
  unsigned __int64 v27; // rdx
  __int64 v28; // r11
  __int64 v29; // r10
  int v30; // eax
  unsigned __int64 *v31; // rsi
  __int64 v32; // r11
  __int64 v33; // r10
  unsigned int v34; // eax
  _QWORD *v35; // rcx
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // r10
  unsigned __int64 v38; // rdx
  __int64 v39; // [rsp+8h] [rbp-108h]
  __int64 v40; // [rsp+10h] [rbp-100h]
  __int64 v42; // [rsp+18h] [rbp-F8h]
  __int64 v43; // [rsp+40h] [rbp-D0h]
  __int64 v44; // [rsp+40h] [rbp-D0h]
  __int64 v45; // [rsp+40h] [rbp-D0h]
  __int64 v46; // [rsp+40h] [rbp-D0h]
  unsigned int v47; // [rsp+48h] [rbp-C8h]
  __int64 v48; // [rsp+48h] [rbp-C8h]
  __int64 v49; // [rsp+48h] [rbp-C8h]
  __int64 v50; // [rsp+48h] [rbp-C8h]
  char v51; // [rsp+5Eh] [rbp-B2h] BYREF
  char v52; // [rsp+5Fh] [rbp-B1h] BYREF
  int v53; // [rsp+60h] [rbp-B0h] BYREF
  int v54; // [rsp+64h] [rbp-ACh] BYREF
  int v55; // [rsp+68h] [rbp-A8h] BYREF
  int v56; // [rsp+6Ch] [rbp-A4h] BYREF
  __int64 v57; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v58; // [rsp+78h] [rbp-98h]
  unsigned __int64 v59; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v60; // [rsp+88h] [rbp-88h]
  __int64 v61; // [rsp+90h] [rbp-80h] BYREF
  unsigned int v62; // [rsp+98h] [rbp-78h]
  __int64 v63; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v64; // [rsp+A8h] [rbp-68h]
  _QWORD *v65; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v66; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v67; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v68; // [rsp+C8h] [rbp-48h]
  unsigned __int64 v69; // [rsp+D0h] [rbp-40h] BYREF
  unsigned int v70; // [rsp+D8h] [rbp-38h]

  v7 = 0;
  if ( a2[2] == 2 )
  {
    LOBYTE(v7) = a4 == -1 || a3 == -1;
    if ( (_BYTE)v7 )
      return 0;
    v10 = *(_QWORD *)a2;
    if ( *(_QWORD *)(*(_QWORD *)a2 + 8LL) == *(_QWORD *)(*(_QWORD *)a2 + 32LL)
      && !(*(_QWORD *)(v10 + 16) + *(_QWORD *)(v10 + 40)) )
    {
      v12 = a5;
      v14 = *(_DWORD *)(**(_QWORD **)(v10 + 24) + 8LL) >> 8;
      v58 = v14;
      if ( v14 <= 0x40 )
      {
        v60 = v14;
        v62 = v14;
        v64 = v14;
        v57 = 0;
        v59 = 0;
        v61 = 0;
        v63 = 0;
      }
      else
      {
        v47 = v14;
        v43 = v10;
        sub_16A4EF0(&v57, 0, 0);
        v60 = v47;
        sub_16A4EF0(&v59, 0, 0);
        v62 = v47;
        sub_16A4EF0(&v61, 0, 0);
        v64 = v47;
        sub_16A4EF0(&v63, 0, 0);
        v12 = a5;
        v10 = v43;
      }
      v51 = 1;
      v15 = *(_QWORD *)(a1 + 8);
      v52 = 1;
      v53 = 0;
      v54 = 0;
      v55 = 0;
      v56 = 0;
      v40 = v12;
      v42 = v10;
      v39 = sub_135E160(*(_QWORD *)v10, &v57, (unsigned int *)&v59, &v53, &v54, v15, 0, a6, a7, &v51, &v52);
      v16 = *(_QWORD *)(v42 + 24);
      v17 = *(_QWORD *)(a1 + 8);
      v51 = 1;
      v52 = 1;
      v18 = sub_135E160(v16, &v61, (unsigned int *)&v63, &v55, &v56, v17, 0, a6, a7, &v51, &v52);
      v19 = v42;
      v20 = v40;
      v21 = v18;
      if ( v58 <= 0x40 )
      {
        if ( v57 != v61 )
          goto LABEL_11;
      }
      else
      {
        v22 = sub_16A5220(&v57, &v61);
        v19 = v42;
        v20 = v40;
        if ( !v22 )
          goto LABEL_11;
      }
      if ( v53 == v55 && v54 == v56 )
      {
        v44 = v20;
        v48 = v19;
        if ( (unsigned __int8)sub_1360E90(a1, v39, v21) )
        {
          v24 = v48;
          v25 = v44;
          v70 = v60;
          if ( v60 > 0x40 )
          {
            sub_16A4FD0(&v69, &v59);
            v25 = v44;
            v24 = v48;
          }
          else
          {
            v69 = v59;
          }
          v45 = v25;
          v49 = v24;
          sub_16A7590(&v69, &v63);
          v26 = v70;
          v27 = v69;
          v28 = v49;
          v29 = v45;
          v66 = v70;
          v65 = (_QWORD *)v69;
          if ( v70 > 0x40 )
          {
            sub_16A4FD0(&v69, &v65);
            v26 = v70;
            v28 = v49;
            v29 = v45;
            if ( v70 > 0x40 )
            {
              sub_16A8F40(&v69);
              v29 = v45;
              v28 = v49;
LABEL_33:
              v46 = v29;
              v50 = v28;
              sub_16A7400(&v69);
              v68 = v70;
              v67 = v69;
              v30 = sub_16A9900(&v65, &v67);
              v31 = &v67;
              v32 = v50;
              v33 = v46;
              if ( v30 < 0 )
                v31 = (unsigned __int64 *)&v65;
              if ( v66 <= 0x40 && (v34 = *((_DWORD *)v31 + 2), v34 <= 0x40) )
              {
                v38 = *v31;
                v66 = *((_DWORD *)v31 + 2);
                v65 = (_QWORD *)(v38 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v34));
              }
              else
              {
                sub_16A51C0(&v65, v31);
                v32 = v50;
                v33 = v46;
                if ( v66 > 0x40 )
                {
                  v35 = (_QWORD *)*v65;
LABEL_39:
                  v36 = (_QWORD)v35 * abs64(*(_QWORD *)(v32 + 16));
                  v37 = abs64(v33);
                  if ( v36 >= v37 + a3 )
                    LOBYTE(v7) = v36 >= v37 + a4;
                  if ( v68 > 0x40 && v67 )
                    j_j___libc_free_0_0(v67);
                  if ( v66 > 0x40 && v65 )
                    j_j___libc_free_0_0(v65);
                  goto LABEL_11;
                }
              }
              v35 = v65;
              goto LABEL_39;
            }
            v27 = v69;
          }
          v69 = (0xFFFFFFFFFFFFFFFFLL >> -v26) & ~v27;
          goto LABEL_33;
        }
      }
LABEL_11:
      if ( v64 > 0x40 && v63 )
        j_j___libc_free_0_0(v63);
      if ( v62 > 0x40 && v61 )
        j_j___libc_free_0_0(v61);
      if ( v60 > 0x40 && v59 )
        j_j___libc_free_0_0(v59);
      if ( v58 > 0x40 && v57 )
        j_j___libc_free_0_0(v57);
    }
  }
  return v7;
}
