// Function: sub_1AECB30
// Address: 0x1aecb30
//
__int64 __fastcall sub_1AECB30(unsigned __int64 a1, unsigned __int8 a2, unsigned int a3, __int64 a4)
{
  int v7; // edx
  int v8; // edx
  __int64 result; // rax
  __int64 v10; // r15
  int v11; // r8d
  __int64 v12; // rdi
  _BYTE *v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rcx
  int v16; // r9d
  unsigned int v17; // r8d
  unsigned __int8 v18; // al
  _QWORD *v19; // rdx
  int v20; // edi
  char v21; // di
  __int64 v22; // rdx
  unsigned __int8 v23; // al
  bool v24; // cl
  unsigned int v25; // ecx
  int v26; // esi
  __int64 *v27; // rax
  __int64 v28; // r12
  _QWORD *v29; // r8
  int v30; // r9d
  __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // r10
  __int64 v36; // r11
  __int64 v37; // rax
  __int64 v38; // rbx
  int v39; // r8d
  int v40; // r9d
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // r15
  _QWORD *v45; // rax
  int v46; // r8d
  int v47; // r9d
  __int64 v48; // r13
  __int64 v49; // rax
  unsigned __int8 v50; // [rsp+1Fh] [rbp-E1h]
  int v51; // [rsp+20h] [rbp-E0h]
  int v52; // [rsp+20h] [rbp-E0h]
  __int64 v53; // [rsp+20h] [rbp-E0h]
  __int64 v54; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v55; // [rsp+28h] [rbp-D8h]
  __int64 v56; // [rsp+28h] [rbp-D8h]
  __int64 v57; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v58; // [rsp+28h] [rbp-D8h]
  __int64 v59; // [rsp+28h] [rbp-D8h]
  __int64 v60; // [rsp+28h] [rbp-D8h]
  _QWORD *v61; // [rsp+28h] [rbp-D8h]
  _QWORD *v62; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v63[2]; // [rsp+40h] [rbp-C0h] BYREF
  char v64; // [rsp+50h] [rbp-B0h]
  char v65; // [rsp+51h] [rbp-AFh]
  __int64 v66; // [rsp+60h] [rbp-A0h] BYREF
  int v67; // [rsp+68h] [rbp-98h] BYREF
  __int64 v68; // [rsp+70h] [rbp-90h]
  int *v69; // [rsp+78h] [rbp-88h]
  int *v70; // [rsp+80h] [rbp-80h]
  __int64 v71; // [rsp+88h] [rbp-78h]
  _QWORD *v72; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int64 v73[2]; // [rsp+98h] [rbp-68h] BYREF
  _BYTE v74[32]; // [rsp+A8h] [rbp-58h] BYREF
  unsigned __int8 v75; // [rsp+C8h] [rbp-38h]

  v7 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v7 <= 0x17u )
  {
    result = 0;
    if ( (_BYTE)v7 != 5 )
      return result;
    v8 = *(unsigned __int16 *)(a1 + 18);
  }
  else
  {
    v8 = v7 - 24;
  }
  result = 0;
  if ( v8 == 27 )
  {
    result = a3;
    LOBYTE(result) = a2 | a3;
    if ( a2 | (unsigned __int8)a3 )
    {
      v10 = *(_QWORD *)a1;
      result = 0;
      if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 11 )
      {
        v11 = *(_DWORD *)(v10 + 8) >> 8;
        if ( *(_DWORD *)(v10 + 8) <= 0x80FFu )
        {
          v12 = *(_QWORD *)(a1 + 8);
          v54 = *(_QWORD *)a1;
          if ( v12 )
          {
            if ( !*(_QWORD *)(v12 + 8) )
            {
              v51 = *(_DWORD *)(v10 + 8) >> 8;
              v13 = sub_1648700(v12);
              v11 = v51;
              if ( v13[16] == 60 )
              {
                v54 = *(_QWORD *)v13;
                v11 = *(_DWORD *)(*(_QWORD *)v13 + 8LL) >> 8;
              }
            }
          }
          v70 = &v67;
          v52 = v11;
          v67 = 0;
          v68 = 0;
          v69 = &v67;
          v71 = 0;
          v14 = sub_1AE8380(a1, a3, &v66, 0);
          v17 = v52;
          v50 = *((_BYTE *)v14 + 56);
          v75 = v50;
          if ( !v50
            || (v19 = (_QWORD *)*v14,
                v73[0] = (unsigned __int64)v74,
                v73[1] = 0x2000000000LL,
                v20 = *((_DWORD *)v14 + 4),
                v72 = v19,
                v20)
            && (sub_1AE7820((__int64)v73, (__int64)(v14 + 1), (__int64)v19, v15, v52, v16), v17 = v52, !v75) )
          {
            v18 = 0;
LABEL_13:
            v55 = v18;
            sub_1AE7A30(v68);
            return v55;
          }
          v21 = (v17 & 0xF) == 0;
          if ( v17 )
          {
            v22 = 0;
            v23 = v50;
            do
            {
              v25 = *(char *)(v73[0] + v22);
              if ( (((unsigned __int8)v22 ^ *(_BYTE *)(v73[0] + v22)) & 7) != 0 )
                v21 = 0;
              else
                v21 &= (v17 >> 3) - 1 - ((unsigned int)v22 >> 3) == v25 >> 3;
              v24 = v17 - 1 - (_DWORD)v22++ == v25;
              v23 &= v24;
            }
            while ( v17 != v22 );
          }
          else
          {
            v23 = v50;
          }
          if ( !a2 || (v26 = 6, !v21) )
          {
            v18 = a3 & v23;
            if ( !v18 )
              goto LABEL_39;
            v26 = 5;
          }
          if ( v54 == v10 )
          {
            v63[0] = v54;
            v42 = (__int64 *)sub_15F2050(a1);
            v43 = sub_15E26F0(v42, v26, v63, 1);
            v65 = 1;
            v44 = v43;
            v64 = 3;
            v63[0] = (__int64)"rev";
            v59 = *(_QWORD *)(*(_QWORD *)v43 + 24LL);
            v45 = sub_1648AB0(72, 2u, 0);
            v48 = (__int64)v45;
            if ( v45 )
            {
              sub_15F1EA0((__int64)v45, **(_QWORD **)(v59 + 16), 54, (__int64)(v45 - 6), 2, a1);
              *(_QWORD *)(v48 + 56) = 0;
              sub_15F5B40(v48, v59, v44, (__int64 *)&v72, 1, (__int64)v63, 0, 0);
            }
            v49 = *(unsigned int *)(a4 + 8);
            if ( (unsigned int)v49 >= *(_DWORD *)(a4 + 12) )
            {
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v46, v47);
              v49 = *(unsigned int *)(a4 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v49) = v48;
            ++*(_DWORD *)(a4 + 8);
          }
          else
          {
            v63[0] = v54;
            v27 = (__int64 *)sub_15F2050(a1);
            v28 = sub_15E26F0(v27, v26, v63, 1);
            v62 = v72;
            if ( v54 != *v72 )
            {
              v65 = 1;
              v63[0] = (__int64)"trunc";
              v64 = 3;
              v29 = (_QWORD *)sub_15FDBD0(36, (__int64)v72, v54, (__int64)v63, a1);
              v31 = *(unsigned int *)(a4 + 8);
              if ( (unsigned int)v31 >= *(_DWORD *)(a4 + 12) )
              {
                v61 = v29;
                sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, (int)v29, v30);
                v31 = *(unsigned int *)(a4 + 8);
                v29 = v61;
              }
              *(_QWORD *)(*(_QWORD *)a4 + 8 * v31) = v29;
              ++*(_DWORD *)(a4 + 8);
              v62 = v29;
            }
            v65 = 1;
            v63[0] = (__int64)"rev";
            v64 = 3;
            v56 = *(_QWORD *)(*(_QWORD *)v28 + 24LL);
            v32 = sub_1648AB0(72, 2u, 0);
            v35 = (__int64)v32;
            if ( v32 )
            {
              v36 = v56;
              v57 = (__int64)v32;
              v53 = v36;
              sub_15F1EA0((__int64)v32, **(_QWORD **)(v36 + 16), 54, (__int64)(v32 - 6), 2, a1);
              *(_QWORD *)(v57 + 56) = 0;
              sub_15F5B40(v57, v53, v28, (__int64 *)&v62, 1, (__int64)v63, 0, 0);
              v35 = v57;
            }
            v37 = *(unsigned int *)(a4 + 8);
            if ( (unsigned int)v37 >= *(_DWORD *)(a4 + 12) )
            {
              v60 = v35;
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v33, v34);
              v37 = *(unsigned int *)(a4 + 8);
              v35 = v60;
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v37) = v35;
            ++*(_DWORD *)(a4 + 8);
            v65 = 1;
            v63[0] = (__int64)"zext";
            v64 = 3;
            v38 = sub_15FDBD0(37, v35, v10, (__int64)v63, a1);
            v41 = *(unsigned int *)(a4 + 8);
            if ( (unsigned int)v41 >= *(_DWORD *)(a4 + 12) )
            {
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v39, v40);
              v41 = *(unsigned int *)(a4 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v41) = v38;
            ++*(_DWORD *)(a4 + 8);
          }
          v18 = v75;
          if ( !v75 )
          {
            v18 = v50;
            goto LABEL_13;
          }
LABEL_39:
          if ( (_BYTE *)v73[0] != v74 )
          {
            v58 = v18;
            _libc_free(v73[0]);
            v18 = v58;
          }
          goto LABEL_13;
        }
      }
    }
  }
  return result;
}
