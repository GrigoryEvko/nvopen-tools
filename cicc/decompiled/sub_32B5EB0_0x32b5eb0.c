// Function: sub_32B5EB0
// Address: 0x32b5eb0
//
__int64 __fastcall sub_32B5EB0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r13d
  __int64 v10; // rax
  __int64 result; // rax
  int v12; // r11d
  int v13; // r10d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  int v21; // r9d
  __int64 v22; // rdx
  int v23; // ecx
  __int64 v24; // rsi
  int v25; // edi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int128 *v33; // rax
  __int128 *v34; // rbx
  __int64 v35; // r14
  __int64 v36; // rdx
  __int64 v37; // r15
  __int64 v38; // r12
  __int128 v39; // rax
  int v40; // r9d
  __int128 v41; // [rsp-30h] [rbp-120h]
  int v42; // [rsp+0h] [rbp-F0h]
  int v43; // [rsp+18h] [rbp-D8h]
  __int64 v44; // [rsp+18h] [rbp-D8h]
  int v45; // [rsp+20h] [rbp-D0h]
  __int64 v46; // [rsp+28h] [rbp-C8h]
  __int64 v47; // [rsp+30h] [rbp-C0h]
  __int64 v48; // [rsp+38h] [rbp-B8h]
  __int64 *v49; // [rsp+38h] [rbp-B8h]
  int v50; // [rsp+40h] [rbp-B0h]
  __int128 *v51; // [rsp+40h] [rbp-B0h]
  __int64 v52; // [rsp+40h] [rbp-B0h]
  int v53; // [rsp+48h] [rbp-A8h]
  int v54; // [rsp+48h] [rbp-A8h]
  __int64 *v55; // [rsp+48h] [rbp-A8h]
  __int64 v56; // [rsp+50h] [rbp-A0h]
  __int64 v57; // [rsp+50h] [rbp-A0h]
  __int128 v58; // [rsp+50h] [rbp-A0h]
  __int64 v59; // [rsp+50h] [rbp-A0h]
  __int64 v60; // [rsp+60h] [rbp-90h] BYREF
  int v61; // [rsp+68h] [rbp-88h]
  __int64 v62; // [rsp+70h] [rbp-80h] BYREF
  int v63; // [rsp+78h] [rbp-78h]
  __int64 v64; // [rsp+80h] [rbp-70h] BYREF
  int v65; // [rsp+88h] [rbp-68h]
  __int64 v66; // [rsp+90h] [rbp-60h] BYREF
  int v67; // [rsp+98h] [rbp-58h]
  __int64 v68; // [rsp+A0h] [rbp-50h] BYREF
  int v69; // [rsp+A8h] [rbp-48h]
  __int64 v70; // [rsp+B0h] [rbp-40h] BYREF
  int v71; // [rsp+B8h] [rbp-38h]

  v7 = a6;
  v10 = *(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5;
  if ( !*((_BYTE *)a1 + 33) && (*(_DWORD *)(a2 + 24) == 51 || *(_DWORD *)(a4 + 24) == 51) )
    return sub_34015B0(
             *a1,
             a6,
             *(unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5),
             *(_QWORD *)(v10 + 8),
             0,
             0);
  v53 = *(unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5);
  v56 = *(_QWORD *)(v10 + 8);
  result = sub_32B51D0(a1, 0, a2, a3, a4, a5, a6);
  v12 = v56;
  v13 = v53;
  if ( result )
    return result;
  if ( *(_DWORD *)(a2 + 24) != 186 || *(_DWORD *)(a4 + 24) != 186 )
    return 0;
  v14 = *(_QWORD *)(a2 + 56);
  if ( v14 && !*(_QWORD *)(v14 + 32) || (v15 = *(_QWORD *)(a4 + 56)) != 0 && !*(_QWORD *)(v15 + 32) )
  {
    v16 = *(_QWORD *)(a2 + 40);
    v22 = *(_QWORD *)(v16 + 40);
    v23 = *(_DWORD *)(v22 + 24);
    if ( v23 != 35 && v23 != 11 )
      goto LABEL_12;
    v17 = *(_QWORD *)(a4 + 40);
    if ( (*(_BYTE *)(v22 + 32) & 8) != 0 )
      goto LABEL_13;
    v24 = *(_QWORD *)(v17 + 40);
    v25 = *(_DWORD *)(v24 + 24);
    if ( v25 != 35 && v25 != 11 )
      goto LABEL_13;
    if ( (*(_BYTE *)(v24 + 32) & 8) != 0 )
      goto LABEL_13;
    v43 = v53;
    v45 = v56;
    v59 = *a1;
    v49 = (__int64 *)(*(_QWORD *)(v22 + 96) + 24LL);
    v55 = (__int64 *)(*(_QWORD *)(v24 + 96) + 24LL);
    sub_9865C0((__int64)&v60, (__int64)v49);
    sub_987160((__int64)&v60, (__int64)v49, v26, v27, v28);
    v63 = v61;
    v61 = 0;
    v62 = v60;
    sub_325F530(&v62, v55);
    v29 = v63;
    v63 = 0;
    v65 = v29;
    v64 = v62;
    if ( (unsigned __int8)sub_33DD210(v59, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), &v64, 0) )
    {
      v42 = v43;
      v44 = *a1;
      sub_9865C0((__int64)&v66, (__int64)v55);
      sub_987160((__int64)&v66, (__int64)v55, v30, v31, v32);
      v69 = v67;
      v67 = 0;
      v68 = v66;
      sub_325F530(&v68, v49);
      v71 = v69;
      v69 = 0;
      v70 = v68;
      LOBYTE(v44) = sub_33DD210(v44, **(_QWORD **)(a4 + 40), *(_QWORD *)(*(_QWORD *)(a4 + 40) + 8LL), &v70, 0);
      sub_969240(&v70);
      sub_969240(&v68);
      sub_969240(&v66);
      sub_969240(&v64);
      sub_969240(&v62);
      sub_969240(&v60);
      v12 = v45;
      v13 = v42;
      if ( (_BYTE)v44 )
      {
        v33 = *(__int128 **)(a4 + 40);
        v34 = *(__int128 **)(a2 + 40);
        v46 = *a1;
        v51 = v33;
        sub_3285E70((__int64)&v70, a2);
        v35 = sub_3406EB0(v46, 187, (unsigned int)&v70, v42, v45, v46, *v34, *v51);
        v37 = v36;
        sub_9C6650(&v70);
        v38 = *a1;
        sub_9865C0((__int64)&v68, (__int64)v49);
        sub_325F510(&v68, v55);
        v71 = v69;
        v70 = v68;
        v69 = 0;
        *(_QWORD *)&v39 = sub_34007B0(v38, (unsigned int)&v70, v7, v42, v45, 0, 0);
        *((_QWORD *)&v41 + 1) = v37;
        *(_QWORD *)&v41 = v35;
        v52 = sub_3406EB0(v38, 186, v7, v42, v45, v40, v41, v39);
        sub_969240(&v70);
        sub_969240(&v68);
        return v52;
      }
    }
    else
    {
      sub_969240(&v64);
      sub_969240(&v62);
      sub_969240(&v60);
      v12 = v45;
      v13 = v43;
    }
    if ( *(_DWORD *)(a2 + 24) != 186 || *(_DWORD *)(a4 + 24) != 186 )
      return 0;
  }
  v16 = *(_QWORD *)(a2 + 40);
LABEL_12:
  v17 = *(_QWORD *)(a4 + 40);
LABEL_13:
  if ( *(_QWORD *)v16 != *(_QWORD *)v17 || *(_DWORD *)(v16 + 8) != *(_DWORD *)(v17 + 8) )
    return 0;
  v18 = *(_QWORD *)(a2 + 56);
  if ( !v18 || *(_QWORD *)(v18 + 32) )
  {
    v19 = *(_QWORD *)(a4 + 56);
    if ( !v19 || *(_QWORD *)(v19 + 32) )
      return 0;
  }
  v47 = v17;
  v48 = v16;
  v50 = v13;
  v54 = v12;
  v57 = *a1;
  sub_3285E70((__int64)&v70, a2);
  *(_QWORD *)&v58 = sub_3406EB0(
                      v57,
                      187,
                      (unsigned int)&v70,
                      v50,
                      v54,
                      v57,
                      *(_OWORD *)(v48 + 40),
                      *(_OWORD *)(v47 + 40));
  *((_QWORD *)&v58 + 1) = v20;
  sub_9C6650(&v70);
  return sub_3406EB0(*a1, 186, v7, v50, v54, v21, *(_OWORD *)*(_QWORD *)(a2 + 40), v58);
}
