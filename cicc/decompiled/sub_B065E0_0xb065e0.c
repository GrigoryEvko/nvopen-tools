// Function: sub_B065E0
// Address: 0xb065e0
//
__int64 __fastcall sub_B065E0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned int a9,
        __int64 a10,
        unsigned int a11,
        __int64 a12,
        int a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        unsigned __int64 a18,
        __int64 a19,
        __int64 a20,
        __int64 a21,
        __int64 a22,
        __int64 a23,
        __int64 a24,
        int a25,
        unsigned int a26,
        char a27)
{
  __int64 v28; // r13
  _QWORD *v29; // r12
  __int64 v30; // rbx
  unsigned int v31; // r15d
  __int64 v32; // r10
  int v33; // eax
  int v34; // r11d
  int v35; // r13d
  unsigned int i; // r12d
  __int64 *v37; // rbx
  __int64 v38; // rsi
  __int64 *v39; // rdx
  __int64 result; // rax
  __int64 v41; // r13
  __int64 v42; // r14
  unsigned int v43; // r12d
  __int64 v44; // [rsp+8h] [rbp-128h]
  __int64 v45; // [rsp+18h] [rbp-118h]
  int v46; // [rsp+20h] [rbp-110h]
  __int64 v47; // [rsp+20h] [rbp-110h]
  __int64 v48; // [rsp+28h] [rbp-108h]
  int v49; // [rsp+30h] [rbp-100h]
  __int16 v51; // [rsp+3Ch] [rbp-F4h]
  __int64 v52; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v54; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v56; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v58; // [rsp+80h] [rbp-B0h]
  __int64 v59; // [rsp+88h] [rbp-A8h]
  unsigned __int64 v60; // [rsp+90h] [rbp-A0h]
  __int64 v61; // [rsp+98h] [rbp-98h] BYREF
  __int64 v62; // [rsp+A0h] [rbp-90h]
  __int64 v63; // [rsp+A8h] [rbp-88h]
  __int64 v64; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v65; // [rsp+B8h] [rbp-78h]
  __int64 v66; // [rsp+C0h] [rbp-70h]
  __int64 v67; // [rsp+C8h] [rbp-68h]
  __int64 v68; // [rsp+D0h] [rbp-60h]
  __int64 v69; // [rsp+D8h] [rbp-58h]
  __int64 v70; // [rsp+E0h] [rbp-50h]
  __int64 v71[2]; // [rsp+E8h] [rbp-48h] BYREF
  int v72; // [rsp+F8h] [rbp-38h]

  v28 = a4;
  v29 = a1;
  v30 = a3;
  v31 = a26;
  v51 = a2;
  if ( a26 )
    goto LABEL_10;
  v32 = *a1;
  v53 = a3;
  v54 = a4;
  LODWORD(v52) = a2;
  v56 = a6;
  LODWORD(v55) = a5;
  v57 = a7;
  v58 = a8;
  v59 = a10;
  v60 = __PAIR64__(a11, a9);
  v61 = a12;
  LODWORD(v62) = a13;
  v63 = a15;
  v64 = a16;
  v65 = a17;
  v66 = a18;
  v67 = a19;
  v44 = v32;
  v68 = a20;
  v69 = a21;
  v70 = a22;
  v71[0] = a23;
  v71[1] = a24;
  v72 = a25;
  v46 = *(_DWORD *)(v32 + 976);
  v48 = *(_QWORD *)(v32 + 960);
  if ( !v46 )
    goto LABEL_9;
  v33 = sub_AFADE0(&v53, &v54, (int *)&v55, &v57, &v56, &v61, &v64, v71);
  v34 = v46;
  v47 = v30;
  v45 = v28;
  v35 = 1;
  v49 = v34 - 1;
  for ( i = (v34 - 1) & v33; ; i = v49 & v43 )
  {
    v37 = (__int64 *)(v48 + 8LL * i);
    v38 = *v37;
    if ( *v37 == -8192 )
      goto LABEL_15;
    if ( v38 == -4096 )
      goto LABEL_17;
    if ( sub_AF52E0((int *)&v52, v38) )
      break;
    v38 = *v37;
LABEL_15:
    if ( v38 == -4096 )
    {
LABEL_17:
      v29 = a1;
      v30 = v47;
      v28 = v45;
      v31 = 0;
      goto LABEL_9;
    }
    v43 = v35 + i;
    ++v35;
  }
  v39 = (__int64 *)(v48 + 8LL * i);
  v29 = a1;
  v30 = v47;
  v28 = v45;
  v31 = 0;
  if ( v39 == (__int64 *)(*(_QWORD *)(v44 + 960) + 8LL * *(unsigned int *)(v44 + 976)) || (result = *v39) == 0 )
  {
LABEL_9:
    result = 0;
    if ( a27 )
    {
LABEL_10:
      v52 = v28;
      v53 = a6;
      v55 = a7;
      v54 = v30;
      v56 = a12;
      v57 = a15;
      v58 = a16;
      v59 = a17;
      v60 = a18;
      v61 = a19;
      v62 = a20;
      v63 = a21;
      v64 = a22;
      v65 = a23;
      v66 = a24;
      v41 = *v29 + 952LL;
      v42 = sub_B97910(56, 15, v31);
      if ( v42 )
      {
        sub_B971C0(v42, (_DWORD)v29, 14, v31, (unsigned int)&v52, 15, 0, 0);
        *(_WORD *)(v42 + 2) = v51;
        *(_DWORD *)(v42 + 16) = a5;
        *(_DWORD *)(v42 + 20) = a11;
        *(_QWORD *)(v42 + 24) = a8;
        *(_DWORD *)(v42 + 4) = a9;
        *(_QWORD *)(v42 + 32) = a10;
        *(_DWORD *)(v42 + 40) = a25;
        *(_DWORD *)(v42 + 44) = a13;
        *(_QWORD *)(v42 + 48) = a14;
      }
      return sub_B06400(v42, v31, v41);
    }
  }
  return result;
}
