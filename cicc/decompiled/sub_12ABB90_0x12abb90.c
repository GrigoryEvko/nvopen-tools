// Function: sub_12ABB90
// Address: 0x12abb90
//
__int64 __fastcall sub_12ABB90(__int64 a1, __int64 a2, unsigned int a3, unsigned __int64 *a4, char a5, char a6)
{
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v20; // rax
  char *v21; // r8
  __int64 v22; // r10
  _QWORD *v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  int v26; // eax
  int v27; // r8d
  __int64 v28; // r10
  __int64 v29; // rdi
  __int64 *v30; // r13
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rsi
  char *v36; // r15
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r13
  __int64 v46; // rax
  __int64 v47; // [rsp+8h] [rbp-D8h]
  __int64 v48; // [rsp+10h] [rbp-D0h]
  char *v49; // [rsp+18h] [rbp-C8h]
  int v50; // [rsp+18h] [rbp-C8h]
  __int64 v51; // [rsp+18h] [rbp-C8h]
  __int64 v52; // [rsp+18h] [rbp-C8h]
  __int64 v53; // [rsp+18h] [rbp-C8h]
  char *v54; // [rsp+20h] [rbp-C0h]
  __int64 v55; // [rsp+20h] [rbp-C0h]
  __int64 v56; // [rsp+20h] [rbp-C0h]
  __int64 v57; // [rsp+20h] [rbp-C0h]
  __int64 v58; // [rsp+28h] [rbp-B8h]
  __int64 v59; // [rsp+28h] [rbp-B8h]
  __int64 v62; // [rsp+38h] [rbp-A8h]
  __int64 v63; // [rsp+48h] [rbp-98h] BYREF
  _QWORD v64[4]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v65; // [rsp+70h] [rbp-70h] BYREF
  __int64 v66; // [rsp+78h] [rbp-68h]
  __int16 v67; // [rsp+80h] [rbp-60h]
  _BYTE v68[16]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v69; // [rsp+A0h] [rbp-40h]

  v10 = (__int64 *)(a2 + 48);
  v11 = sub_1643350(*(_QWORD *)(a2 + 40));
  v12 = sub_159C470(v11, a3, 0);
  v13 = *(_QWORD *)(a4[9] + 16);
  if ( a6 )
  {
    v14 = *(_QWORD *)(v13 + 16);
    v49 = sub_128F980(a2, v13);
    v54 = sub_128F980(a2, v14);
    v48 = sub_15A06D0(*(_QWORD *)v54);
    v15 = sub_126A190(*(_QWORD **)(a2 + 32), 5301, 0, 0);
    v64[1] = v12;
    v58 = v15;
    v64[0] = v49;
    v67 = 257;
    if ( (unsigned __int8)v54[16] > 0x10u || *(_BYTE *)(v48 + 16) > 0x10u )
    {
      v69 = 257;
      v20 = sub_1648A60(56, 2);
      v21 = v54;
      v22 = v20;
      if ( v20 )
      {
        v55 = v20;
        v23 = *(_QWORD **)v21;
        v47 = v22;
        v50 = (int)v21;
        if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) == 16 )
        {
          v24 = v23[4];
          v25 = sub_1643320(*v23);
          v26 = sub_16463B0(v25, (unsigned int)v24);
          v27 = v50;
          v28 = v47;
        }
        else
        {
          v26 = sub_1643320(*v23);
          v28 = v47;
          v27 = v50;
        }
        v51 = v28;
        sub_15FEC10(v28, v26, 51, 33, v27, v48, (__int64)v68, 0);
        v22 = v51;
      }
      else
      {
        v55 = 0;
      }
      v29 = *(_QWORD *)(a2 + 56);
      if ( v29 )
      {
        v30 = *(__int64 **)(a2 + 64);
        v52 = v22;
        sub_157E9D0(v29 + 40, v22);
        v22 = v52;
        v31 = *v30;
        v32 = *(_QWORD *)(v52 + 24);
        *(_QWORD *)(v52 + 32) = v30;
        v31 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v52 + 24) = v31 | v32 & 7;
        *(_QWORD *)(v31 + 8) = v52 + 24;
        *v30 = *v30 & 7 | (v52 + 24);
      }
      v53 = v22;
      sub_164B780(v55, &v65);
      v33 = *(_QWORD *)(a2 + 48);
      v16 = v53;
      if ( v33 )
      {
        v63 = *(_QWORD *)(a2 + 48);
        sub_1623A60(&v63, v33, 2);
        v16 = v53;
        v34 = v53 + 48;
        if ( *(_QWORD *)(v53 + 48) )
        {
          sub_161E7C0(v53 + 48);
          v16 = v53;
          v34 = v53 + 48;
        }
        v35 = v63;
        *(_QWORD *)(v16 + 48) = v63;
        if ( v35 )
        {
          v56 = v16;
          sub_1623210(&v63, v35, v34);
          v16 = v56;
        }
      }
    }
    else
    {
      v16 = sub_15A37B0(33, v54, v48, 0);
    }
    v64[2] = v16;
    v69 = 257;
    v17 = sub_1285290(v10, *(_QWORD *)(v58 + 24), v58, (int)v64, 3, (__int64)v68, 0);
    v69 = 257;
    if ( a5 )
    {
      LODWORD(v65) = 0;
      v18 = sub_12A9E60(v10, v17, (__int64)&v65, 1, (__int64)v68);
    }
    else
    {
      LODWORD(v65) = 1;
      v42 = sub_12A9E60(v10, v17, (__int64)&v65, 1, (__int64)v68);
      v43 = sub_127A030(*(_QWORD *)(a2 + 32) + 8LL, *a4, 0);
      v69 = 257;
      v18 = sub_12AA3B0(v10, 0x25u, v42, v43, (__int64)v68);
    }
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = v18;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
  }
  else
  {
    v36 = sub_128F980(a2, v13);
    v37 = sub_15A06D0(*(_QWORD *)v36);
    if ( a5 )
    {
      v62 = v37;
      v38 = sub_126A190(*(_QWORD **)(a2 + 32), 5293, 0, 0);
      v65 = v12;
      v39 = v38;
      v69 = 257;
      v66 = sub_12AA0C0(v10, 0x21u, v36, v62, (__int64)v68);
      v69 = 257;
      v40 = sub_1285290(v10, *(_QWORD *)(v39 + 24), v39, (int)&v65, 2, (__int64)v68, 0);
      LODWORD(v64[0]) = 0;
      v69 = 257;
      v41 = sub_12A9E60(v10, v40, (__int64)v64, 1, (__int64)v68);
    }
    else
    {
      v57 = v37;
      v44 = sub_126A190(*(_QWORD **)(a2 + 32), 5300, 0, 0);
      v69 = 257;
      v59 = v44;
      v65 = v12;
      v66 = sub_12AA0C0(v10, 0x21u, v36, v57, (__int64)v68);
      v69 = 257;
      v45 = sub_1285290(v10, *(_QWORD *)(v59 + 24), v59, (int)&v65, 2, (__int64)v68, 0);
      v46 = sub_127A030(*(_QWORD *)(a2 + 32) + 8LL, *a4, 0);
      v69 = 257;
      v41 = sub_12AA3B0(v10, 0x25u, v45, v46, (__int64)v68);
    }
    *(_QWORD *)a1 = v41;
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
  }
  return a1;
}
