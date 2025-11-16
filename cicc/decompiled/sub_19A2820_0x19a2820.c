// Function: sub_19A2820
// Address: 0x19a2820
//
void __fastcall sub_19A2820(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        char a9)
{
  __int64 v12; // rax
  __int64 *v13; // r12
  __int64 v14; // rdx
  __int64 *v15; // rdi
  __int64 v16; // r8
  unsigned int v17; // ecx
  __int64 v18; // rsi
  unsigned int v19; // r9d
  char v20; // al
  char v21; // dl
  __int64 v22; // r15
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rdx
  int v29; // r8d
  int v30; // r9d
  __int64 v31; // r12
  bool v32; // al
  int v33; // r8d
  int v34; // r9d
  char v35; // dl
  int v36; // esi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  unsigned int v40; // ecx
  __int64 *v41; // rdi
  unsigned int v42; // r9d
  __int64 v43; // rsi
  __int64 v44; // r8
  int v45; // r8d
  int v46; // r9d
  __int64 *v47; // rcx
  __int64 *v48; // rdx
  int v49; // eax
  __int64 v50; // rax
  int v51; // [rsp-8h] [rbp-128h]
  __int64 v52; // [rsp+8h] [rbp-118h]
  __int64 v54; // [rsp+20h] [rbp-100h]
  __int64 v56; // [rsp+40h] [rbp-E0h]
  __int64 *v57; // [rsp+40h] [rbp-E0h]
  __int64 v58; // [rsp+40h] [rbp-E0h]
  __int64 v59; // [rsp+48h] [rbp-D8h]
  __int64 v60; // [rsp+68h] [rbp-B8h] BYREF
  __int64 v61[2]; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD v62[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v63; // [rsp+90h] [rbp-90h] BYREF
  __int64 v64; // [rsp+98h] [rbp-88h]
  char v65; // [rsp+A0h] [rbp-80h]
  __int64 v66; // [rsp+A8h] [rbp-78h]
  _BYTE *v67; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v68; // [rsp+B8h] [rbp-68h]
  _BYTE v69[32]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v70; // [rsp+E0h] [rbp-40h]
  __int64 v71; // [rsp+E8h] [rbp-38h]

  if ( a9 )
    v12 = a4[10];
  else
    v12 = *(_QWORD *)(a4[4] + 8 * a6);
  v60 = v12;
  v13 = *(__int64 **)a5;
  v59 = *(_QWORD *)a5 + 8LL * *(unsigned int *)(a5 + 8);
  if ( v59 != *(_QWORD *)a5 )
  {
    v52 = 8 * a6;
    v54 = (__int64)(a4 + 4);
    do
    {
      v21 = *((_BYTE *)a4 + 16);
      v22 = *v13;
      v23 = *((_DWORD *)a4 + 10);
      v63 = *a4;
      v65 = v21;
      v24 = a4[1];
      v25 = a4[3];
      v67 = v69;
      v64 = v24;
      v66 = v25;
      v68 = 0x400000000LL;
      if ( v23 )
      {
        sub_19930D0((__int64)&v67, v54, v25, 0x400000000LL, a5, v23);
        v24 = a4[1];
      }
      v14 = a4[10];
      v15 = (__int64 *)a1[4];
      v16 = *(_QWORD *)(a2 + 40);
      v17 = *(_DWORD *)(a2 + 32);
      v64 = v24 - v22;
      v18 = *(_QWORD *)(a2 + 712);
      v19 = *(_DWORD *)(a2 + 48);
      v70 = v14;
      v71 = a4[11];
      v20 = sub_1995490(v15, v18 - v22, *(_QWORD *)(a2 + 720) - v22, v17, v16, v19, (__int64)&v63);
      LODWORD(a5) = v51;
      if ( v20 )
      {
        v56 = a1[1];
        v26 = sub_1456040(v60);
        v62[0] = sub_145CF80(v56, v26, v22, 0);
        v61[0] = (__int64)v62;
        v62[1] = v60;
        v61[1] = 0x200000002LL;
        v27 = sub_147DD40(v56, v61, 0, 0, a7, a8);
        v28 = (__int64)v27;
        if ( (_QWORD *)v61[0] != v62 )
        {
          v57 = v27;
          _libc_free(v61[0]);
          v28 = (__int64)v57;
        }
        v58 = v28;
        if ( sub_14560B0(v28) )
        {
          if ( a9 )
          {
            v66 = 0;
            v70 = 0;
          }
          else
          {
            v47 = (__int64 *)&v67[v52];
            v48 = (__int64 *)&v67[8 * (unsigned int)v68 - 8];
            v49 = v68;
            if ( &v67[v52] != (_BYTE *)v48 )
            {
              v50 = *v47;
              *v47 = *v48;
              *v48 = v50;
              v49 = v68;
            }
            LODWORD(v68) = v49 - 1;
          }
          sub_19932F0((__int64)&v63, a1[5]);
        }
        else if ( a9 )
        {
          v70 = v58;
        }
        else
        {
          *(_QWORD *)&v67[v52] = v58;
        }
        sub_19A1660((__int64)a1, a2, a3, (__int64)&v63, v29, v30);
      }
      if ( v67 != v69 )
        _libc_free((unsigned __int64)v67);
      ++v13;
    }
    while ( (__int64 *)v59 != v13 );
  }
  v31 = sub_199D980((__int64)&v60, a1[1], a7, a8);
  v32 = sub_14560B0(v60);
  if ( v31 && !v32 )
  {
    v35 = *((_BYTE *)a4 + 16);
    v36 = *((_DWORD *)a4 + 10);
    v63 = *a4;
    v37 = a4[1];
    v65 = v35;
    v38 = a4[3];
    v67 = v69;
    v64 = v37;
    v66 = v38;
    v68 = 0x400000000LL;
    if ( v36 )
    {
      sub_19930D0((__int64)&v67, (__int64)(a4 + 4), v38, 0x400000000LL, v33, v34);
      v37 = v64;
    }
    v39 = a4[10];
    v40 = *(_DWORD *)(a2 + 32);
    v41 = (__int64 *)a1[4];
    v42 = *(_DWORD *)(a2 + 48);
    v64 = v31 + v37;
    v43 = *(_QWORD *)(a2 + 712);
    v44 = *(_QWORD *)(a2 + 40);
    v70 = v39;
    v71 = a4[11];
    if ( sub_1995490(v41, v43, *(_QWORD *)(a2 + 720), v40, v44, v42, (__int64)&v63) )
    {
      if ( a9 )
        v70 = v60;
      else
        *(_QWORD *)&v67[8 * a6] = v60;
      sub_19A1660((__int64)a1, a2, a3, (__int64)&v63, v45, v46);
    }
    if ( v67 != v69 )
      _libc_free((unsigned __int64)v67);
  }
}
