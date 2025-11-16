// Function: sub_264E850
// Address: 0x264e850
//
__int64 __fastcall sub_264E850(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 **v5; // rax
  __int64 **v6; // rdx
  __int64 **v7; // rcx
  __int64 *v8; // rax
  __int64 *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // rax
  int v14; // edi
  __int64 v15; // r8
  int v16; // esi
  __int64 v17; // r9
  int v18; // ecx
  volatile signed __int32 *v19; // rdi
  volatile signed __int32 *v20; // r13
  __int64 v21; // r15
  char v22; // al
  __int64 v23; // rax
  int v24; // edi
  int v25; // esi
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // ecx
  __int64 i; // r12
  __int64 v31; // r12
  int v32; // eax
  unsigned int v33; // eax
  __int64 v34; // rbx
  _DWORD *v35; // r14
  _DWORD *v36; // r13
  __int64 *v39; // [rsp+10h] [rbp-140h]
  __int64 **v41; // [rsp+20h] [rbp-130h]
  __int64 *v43; // [rsp+40h] [rbp-110h]
  unsigned __int8 v44; // [rsp+48h] [rbp-108h]
  __int64 v45; // [rsp+48h] [rbp-108h]
  char v46; // [rsp+48h] [rbp-108h]
  __int64 v47; // [rsp+48h] [rbp-108h]
  __int64 v48; // [rsp+48h] [rbp-108h]
  __int64 *v49; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v50; // [rsp+60h] [rbp-F0h] BYREF
  volatile signed __int32 *v51; // [rsp+68h] [rbp-E8h]
  _QWORD v52[2]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v53; // [rsp+80h] [rbp-D0h]
  __int64 v54; // [rsp+88h] [rbp-C8h]
  _QWORD v55[3]; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+A8h] [rbp-A8h]
  __int64 v57; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v58; // [rsp+B8h] [rbp-98h]
  __int64 v59; // [rsp+C0h] [rbp-90h]
  __int64 v60; // [rsp+C8h] [rbp-88h]
  __int64 v61; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v62; // [rsp+D8h] [rbp-78h]
  _DWORD *v63; // [rsp+E0h] [rbp-70h]
  _DWORD *v64; // [rsp+E8h] [rbp-68h]
  __int64 v65; // [rsp+F0h] [rbp-60h] BYREF
  volatile signed __int32 *v66; // [rsp+F8h] [rbp-58h]
  __int64 v67; // [rsp+100h] [rbp-50h]
  __int64 v68; // [rsp+108h] [rbp-48h]
  char v69; // [rsp+110h] [rbp-40h]

  v5 = (__int64 **)(a3 + 48);
  v6 = (__int64 **)(a3 + 72);
  if ( !a4 )
    v5 = v6;
  v7 = v5;
  v41 = v5;
  v8 = *v5;
  v9 = v7[1];
  v52[0] = 0;
  v52[1] = 0;
  v53 = 0;
  v54 = 0;
  memset(v55, 0, sizeof(v55));
  v56 = 0;
  v39 = v9;
  if ( (_BYTE)qword_4FF3708 )
  {
    if ( v8 == v9 )
      goto LABEL_28;
    v43 = v8;
    for ( i = 0; ; i = v55[0] )
    {
      v31 = i + 1;
      v32 = *(_DWORD *)(*v43 + 40);
      if ( v32 )
      {
        v33 = sub_AF1560(4 * v32 / 3u + 1);
        v55[0] = v31;
        if ( (unsigned int)v56 < v33 )
          sub_A08C50((__int64)v55, v33);
      }
      else
      {
        v55[0] = v31;
      }
      v34 = *v43;
      sub_22B0690(&v61, (__int64 *)(*v43 + 24));
      v48 = *(_QWORD *)(v34 + 32) + 4LL * *(unsigned int *)(v34 + 48);
      if ( (_DWORD *)v48 != v63 )
      {
        v35 = v64;
        v36 = v63;
        do
        {
          LODWORD(v57) = *v36;
          sub_22B6470((__int64)&v65, (__int64)v55, (int *)&v57);
          if ( !v69 )
            sub_22B6470((__int64)&v65, (__int64)v52, (int *)&v57);
          do
            ++v36;
          while ( v35 != v36 && *v36 > 0xFFFFFFFD );
        }
        while ( v36 != (_DWORD *)v48 );
      }
      v43 += 2;
      if ( v39 == v43 )
        break;
    }
    v8 = *v41;
    v39 = v41[1];
  }
  v49 = v8;
  if ( v8 != v39 )
  {
    while ( 1 )
    {
      v20 = (volatile signed __int32 *)v8[1];
      v21 = *v8;
      if ( v20 )
      {
        if ( &_pthread_key_create )
          _InterlockedAdd(v20 + 2, 1u);
        else
          ++*((_DWORD *)v20 + 2);
      }
      v57 = 0;
      v58 = 0;
      v59 = 0;
      v60 = 0;
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      sub_264C4B0(v21 + 24, a5, (__int64)&v57, (__int64)&v61);
      if ( (_DWORD)v53 )
      {
        sub_264C5E0((__int64)&v65, (__int64)&v57, (__int64)v52);
        sub_2649AE0(a5, (__int64)&v65);
        sub_C7D6A0((__int64)v66, 4LL * (unsigned int)v68, 4);
      }
      else
      {
        v10 = *(_QWORD *)(a5 + 8);
        v11 = v62;
        ++*(_QWORD *)a5;
        *(_QWORD *)(a5 + 8) = v11;
        v62 = v10;
        LODWORD(v10) = *(_DWORD *)(a5 + 16);
        *(_DWORD *)(a5 + 16) = (_DWORD)v63;
        LODWORD(v63) = v10;
        LODWORD(v10) = *(_DWORD *)(a5 + 20);
        *(_DWORD *)(a5 + 20) = HIDWORD(v63);
        HIDWORD(v63) = v10;
        LODWORD(v10) = *(_DWORD *)(a5 + 24);
        ++v61;
        *(_DWORD *)(a5 + 24) = (_DWORD)v64;
        LODWORD(v64) = v10;
      }
      if ( !(_DWORD)v59 )
        goto LABEL_14;
      if ( a4 )
        break;
      v22 = sub_26484B0(a1, (__int64)&v57);
      v65 = 0;
      v46 = v22;
      v23 = sub_22077B0(0x48u);
      if ( v23 )
      {
        v24 = v59;
        *(_QWORD *)(v23 + 8) = 0x100000001LL;
        v25 = HIDWORD(v59);
        *(_QWORD *)v23 = off_49D3C50;
        v26 = v58;
        v27 = *(_QWORD *)(v21 + 8);
        v28 = v60;
        *(_QWORD *)(v23 + 16) = a2;
        *(_DWORD *)(v23 + 56) = v24;
        *(_DWORD *)(v23 + 60) = v25;
        *(_BYTE *)(v23 + 32) = v46;
        *(_QWORD *)(v23 + 24) = v27;
        *(_BYTE *)(v23 + 33) = 0;
        *(_QWORD *)(v23 + 40) = 1;
        *(_QWORD *)(v23 + 48) = v26;
        *(_DWORD *)(v23 + 64) = v28;
        v47 = v23;
        ++v57;
        v58 = 0;
        v59 = 0;
        LODWORD(v60) = 0;
        sub_C7D6A0(0, 0, 4);
        v23 = v47;
      }
      v66 = (volatile signed __int32 *)v23;
      v65 = v23 + 16;
      sub_2647660((unsigned __int64 *)(a2 + 72), &v65);
      sub_2647660((unsigned __int64 *)(*(_QWORD *)(v65 + 8) + 48LL), &v65);
      v19 = v66;
      if ( v66 )
        goto LABEL_12;
      if ( !*(_DWORD *)(v21 + 40) )
      {
LABEL_27:
        sub_264E780(v21, (__int64 *)&v49, a4);
        goto LABEL_15;
      }
LABEL_14:
      v49 += 2;
LABEL_15:
      sub_C7D6A0(v62, 4LL * (unsigned int)v64, 4);
      sub_C7D6A0(v58, 4LL * (unsigned int)v60, 4);
      if ( v20 )
        sub_A191D0(v20);
      v8 = v49;
      if ( v49 == v41[1] )
        goto LABEL_28;
    }
    v12 = sub_26484B0(a1, (__int64)&v57);
    v50 = 0;
    v44 = v12;
    v13 = sub_22077B0(0x48u);
    if ( v13 )
    {
      v14 = v59;
      *(_QWORD *)(v13 + 8) = 0x100000001LL;
      v15 = v58;
      *(_QWORD *)v13 = off_49D3C50;
      v16 = HIDWORD(v59);
      v17 = *(_QWORD *)v21;
      v18 = v60;
      *(_QWORD *)(v13 + 24) = a2;
      *(_DWORD *)(v13 + 56) = v14;
      *(_QWORD *)(v13 + 16) = v17;
      *(_WORD *)(v13 + 32) = v44;
      *(_QWORD *)(v13 + 40) = 1;
      *(_QWORD *)(v13 + 48) = v15;
      *(_DWORD *)(v13 + 60) = v16;
      *(_DWORD *)(v13 + 64) = v18;
      v45 = v13;
      ++v57;
      v58 = 0;
      v59 = 0;
      LODWORD(v60) = 0;
      v65 = 2;
      v66 = 0;
      v67 = 0;
      v68 = 0;
      sub_2342640((__int64)&v65);
      v13 = v45;
    }
    v51 = (volatile signed __int32 *)v13;
    v50 = v13 + 16;
    sub_2647660((unsigned __int64 *)(a2 + 48), &v50);
    sub_2647660((unsigned __int64 *)(*(_QWORD *)v50 + 72LL), &v50);
    v19 = v51;
    if ( v51 )
LABEL_12:
      sub_A191D0(v19);
    if ( !*(_DWORD *)(v21 + 40) )
      goto LABEL_27;
    goto LABEL_14;
  }
LABEL_28:
  sub_2342640((__int64)v55);
  return sub_2342640((__int64)v52);
}
