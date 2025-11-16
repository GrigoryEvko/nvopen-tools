// Function: sub_19172C0
// Address: 0x19172c0
//
__int64 __fastcall sub_19172C0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v6; // r14
  unsigned int v10; // r9d
  int v11; // esi
  __int64 result; // rax
  __int64 v13; // r10
  unsigned int v14; // ecx
  __int64 v15; // rdx
  int v16; // r8d
  __int64 v17; // rcx
  int v18; // esi
  unsigned int v19; // edx
  unsigned int v20; // r14d
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdi
  int v25; // ecx
  int v26; // ecx
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned __int64 v34; // rax
  int v35; // edx
  unsigned int v36; // ecx
  unsigned int v37; // ebx
  char v38; // al
  __int64 v39; // rbx
  unsigned int v40; // esi
  int v41; // eax
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rcx
  int v45; // r8d
  int v46; // r9d
  int v47; // eax
  int v48; // edx
  __int64 v49; // r9
  unsigned int v50; // r8d
  int v51; // r11d
  __int64 v52; // r10
  __int64 v53; // r8
  unsigned int v54; // r14d
  __int64 v55; // rax
  __int64 v56; // r15
  unsigned int v57; // edx
  unsigned int *v58; // rbx
  unsigned int v59; // eax
  unsigned int v60; // edi
  int v61; // [rsp+10h] [rbp-C0h]
  unsigned int v62; // [rsp+10h] [rbp-C0h]
  __int64 v63; // [rsp+10h] [rbp-C0h]
  unsigned int v64; // [rsp+1Ch] [rbp-B4h] BYREF
  unsigned int v65; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+28h] [rbp-A8h]
  char v67; // [rsp+30h] [rbp-A0h]
  _BYTE *v68; // [rsp+38h] [rbp-98h] BYREF
  __int64 v69; // [rsp+40h] [rbp-90h]
  _BYTE v70[24]; // [rsp+48h] [rbp-88h] BYREF
  __int64 v71[2]; // [rsp+60h] [rbp-70h] BYREF
  char v72; // [rsp+70h] [rbp-60h]
  char *v73; // [rsp+78h] [rbp-58h]
  __int64 v74; // [rsp+80h] [rbp-50h]
  char v75; // [rsp+88h] [rbp-48h] BYREF

  v6 = a1 + 120;
  v10 = *(_DWORD *)(a1 + 144);
  v64 = a4;
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_44;
  }
  v11 = a4;
  result = a4;
  v13 = *(_QWORD *)(a1 + 128);
  v14 = (v10 - 1) & (37 * a4);
  v15 = v13 + 16LL * v14;
  v16 = *(_DWORD *)v15;
  if ( v11 != *(_DWORD *)v15 )
  {
    v61 = 1;
    v24 = 0;
    while ( v16 != -1 )
    {
      if ( v16 == -2 && !v24 )
        v24 = v15;
      v14 = (v10 - 1) & (v61 + v14);
      v15 = v13 + 16LL * v14;
      v16 = *(_DWORD *)v15;
      if ( v11 == *(_DWORD *)v15 )
        goto LABEL_3;
      ++v61;
    }
    v25 = *(_DWORD *)(a1 + 136);
    if ( !v24 )
      v24 = v15;
    ++*(_QWORD *)(a1 + 120);
    v26 = v25 + 1;
    if ( 4 * v26 < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 140) - v26 <= v10 >> 3 )
      {
        sub_1910EE0(v6, v10);
        sub_190E640(v6, (int *)&v64, v71);
        v24 = v71[0];
        LODWORD(result) = v64;
        v26 = *(_DWORD *)(a1 + 136) + 1;
      }
LABEL_23:
      *(_DWORD *)(a1 + 136) = v26;
      if ( *(_DWORD *)v24 != -1 )
        --*(_DWORD *)(a1 + 140);
      *(_QWORD *)(v24 + 8) = 0;
      *(_DWORD *)v24 = result;
      v11 = v64;
      goto LABEL_26;
    }
LABEL_44:
    sub_1910EE0(v6, 2 * v10);
    v47 = *(_DWORD *)(a1 + 144);
    if ( !v47 )
    {
      ++*(_DWORD *)(a1 + 136);
      BUG();
    }
    v48 = v47 - 1;
    v49 = *(_QWORD *)(a1 + 128);
    v26 = *(_DWORD *)(a1 + 136) + 1;
    v50 = (v47 - 1) & (37 * v64);
    v24 = v49 + 16LL * v50;
    LODWORD(result) = *(_DWORD *)v24;
    if ( v64 != *(_DWORD *)v24 )
    {
      v51 = 1;
      v52 = 0;
      while ( (_DWORD)result != -1 )
      {
        if ( !v52 && (_DWORD)result == -2 )
          v52 = v24;
        v50 = v48 & (v51 + v50);
        v24 = v49 + 16LL * v50;
        LODWORD(result) = *(_DWORD *)v24;
        if ( v64 == *(_DWORD *)v24 )
          goto LABEL_23;
        ++v51;
      }
      LODWORD(result) = v64;
      if ( v52 )
        v24 = v52;
    }
    goto LABEL_23;
  }
LABEL_3:
  v17 = *(_QWORD *)(v15 + 8);
  if ( v17 )
  {
    v18 = *(_DWORD *)(v17 + 20);
    v19 = v18 & 0xFFFFFFF;
    if ( (v18 & 0xFFFFFFF) != 0 )
    {
      v20 = 0;
      v21 = v17;
      while ( 1 )
      {
        if ( a3 == *(_QWORD *)(v21 + 40) )
        {
          v22 = 8LL * v20 + 24LL * *(unsigned int *)(v21 + 56) + 8;
          if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
          {
            v23 = *(_QWORD *)(v21 - 8);
            if ( a2 != *(_QWORD *)(v23 + v22) )
              goto LABEL_6;
          }
          else
          {
            v23 = v21 - 24LL * v19;
            if ( a2 != *(_QWORD *)(v23 + v22) )
              goto LABEL_6;
          }
          result = sub_190AC30(a1, *(_QWORD *)(v23 + 24LL * v20), 0);
          if ( (_DWORD)result )
            return result;
          v18 = *(_DWORD *)(v21 + 20);
        }
LABEL_6:
        ++v20;
        v19 = v18 & 0xFFFFFFF;
        if ( v20 == (v18 & 0xFFFFFFF) )
          return v64;
      }
    }
    return result;
  }
LABEL_26:
  if ( !(unsigned __int8)sub_1910080(a1, v11, a3, a5) )
    return v64;
  v29 = *(_QWORD *)(a1 + 96);
  result = v64;
  if ( v64 < (unsigned __int64)((*(_QWORD *)(a1 + 104) - v29) >> 2) )
  {
    v30 = *(unsigned int *)(v29 + 4LL * v64);
    if ( (_DWORD)v30 )
    {
      v31 = *(_QWORD *)(a1 + 72) + 56 * v30;
      v65 = *(_DWORD *)v31;
      v66 = *(_QWORD *)(v31 + 8);
      v32 = *(unsigned __int8 *)(v31 + 16);
      v68 = v70;
      v69 = 0x400000000LL;
      v33 = *(unsigned int *)(v31 + 32);
      v67 = v32;
      if ( !(_DWORD)v33 )
      {
LABEL_30:
        if ( (_BYTE)v32 )
        {
          v34 = (unsigned __int64)v68;
          v35 = *(_DWORD *)v68;
          v36 = *((_DWORD *)v68 + 1);
          if ( *(_DWORD *)v68 > v36 )
          {
            *(_DWORD *)v68 = v36;
            v37 = v65;
            *(_DWORD *)(v34 + 4) = v35;
            if ( (v37 >> 8) - 51 <= 1 )
            {
              v60 = (unsigned __int8)v37;
              LOBYTE(v37) = 0;
              v65 = sub_15FF5D0(v60) | v37;
            }
          }
        }
        v38 = sub_190F0D0(a1 + 32, (__int64)&v65, v71);
        v39 = v71[0];
        if ( v38 )
        {
          result = *(unsigned int *)(v71[0] + 56);
          if ( (_DWORD)result )
          {
LABEL_41:
            if ( v68 != v70 )
            {
              v62 = result;
              _libc_free((unsigned __int64)v68);
              return v62;
            }
            return result;
          }
LABEL_40:
          result = v64;
          goto LABEL_41;
        }
        v40 = *(_DWORD *)(a1 + 56);
        v41 = *(_DWORD *)(a1 + 48);
        ++*(_QWORD *)(a1 + 32);
        v42 = v41 + 1;
        if ( 4 * v42 >= 3 * v40 )
        {
          v40 *= 2;
        }
        else if ( v40 - *(_DWORD *)(a1 + 52) - v42 > v40 >> 3 )
        {
LABEL_37:
          *(_DWORD *)(a1 + 48) = v42;
          v73 = &v75;
          LODWORD(v71[0]) = -1;
          v72 = 0;
          v74 = 0x400000000LL;
          if ( !sub_190A670(v39, (__int64)v71) )
            --*(_DWORD *)(a1 + 52);
          *(_DWORD *)v39 = v65;
          *(_QWORD *)(v39 + 8) = v66;
          *(_BYTE *)(v39 + 16) = v67;
          sub_1909410(v39 + 24, (__int64)&v68, v43, v44, v45, v46);
          *(_DWORD *)(v39 + 56) = 0;
          goto LABEL_40;
        }
        sub_1911AB0(a1 + 32, v40);
        sub_190F0D0(a1 + 32, (__int64)&v65, v71);
        v39 = v71[0];
        v42 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_37;
      }
      sub_1909410((__int64)&v68, v31 + 24, v32, v33, v27, v28);
      if ( !(_DWORD)v69 )
      {
LABEL_60:
        LOBYTE(v32) = v67;
        goto LABEL_30;
      }
      v53 = a5;
      v54 = 0;
      v55 = 0;
      v56 = a3;
      while ( 1 )
      {
        if ( v54 > 1 )
        {
          v57 = v65;
          if ( v65 == 63 )
            goto LABEL_56;
        }
        else
        {
          if ( !v54 )
          {
LABEL_55:
            v63 = v53;
            v58 = (unsigned int *)&v68[4 * v55];
            v59 = sub_19170B0(a1, a2, v56, *v58);
            v53 = v63;
            *v58 = v59;
            goto LABEL_56;
          }
          v57 = v65;
        }
        if ( v57 != 62 )
          goto LABEL_55;
LABEL_56:
        v55 = v54 + 1;
        v54 = v55;
        if ( (unsigned int)v55 >= (unsigned int)v69 )
          goto LABEL_60;
      }
    }
  }
  return result;
}
