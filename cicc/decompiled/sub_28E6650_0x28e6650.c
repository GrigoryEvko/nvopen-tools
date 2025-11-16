// Function: sub_28E6650
// Address: 0x28e6650
//
void __fastcall sub_28E6650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v7; // rbx
  char *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 *v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // r12
  unsigned __int64 v20; // rax
  __int64 v21; // r12
  int v22; // eax
  char *v23; // rbx
  unsigned int v24; // r15d
  int v25; // r13d
  __int64 v26; // rsi
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  unsigned __int64 v29; // rax
  int v30; // edx
  unsigned __int64 v31; // rax
  _BYTE *v32; // rsi
  __int64 v33; // r12
  __int64 v34; // r12
  unsigned int v35; // r12d
  bool v36; // r12
  unsigned __int64 v37; // rax
  __int64 v38; // r12
  unsigned int v39; // r12d
  bool v40; // r12
  unsigned __int64 v41; // rax
  __int64 v42; // r12
  __int64 *v43; // r13
  __int64 i; // rbx
  __int64 v45; // r14
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // [rsp+8h] [rbp-138h]
  __int64 v49; // [rsp+18h] [rbp-128h]
  __int64 v50; // [rsp+20h] [rbp-120h]
  __int64 v51; // [rsp+20h] [rbp-120h]
  __int64 *v52; // [rsp+28h] [rbp-118h]
  __int64 v53; // [rsp+28h] [rbp-118h]
  unsigned int v54; // [rsp+28h] [rbp-118h]
  __int64 *v55; // [rsp+30h] [rbp-110h]
  __int64 v56; // [rsp+38h] [rbp-108h]
  unsigned int v57; // [rsp+38h] [rbp-108h]
  __int64 v58; // [rsp+40h] [rbp-100h]
  __int64 *v59; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v60; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v61; // [rsp+58h] [rbp-E8h]
  unsigned __int64 v62; // [rsp+60h] [rbp-E0h] BYREF
  unsigned int v63; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v64; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int v65; // [rsp+78h] [rbp-C8h]
  __int64 *v66; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+88h] [rbp-B8h]
  _BYTE v68[176]; // [rsp+90h] [rbp-B0h] BYREF

  v6 = *(__int64 **)(a2 + 32);
  v49 = *v6;
  v66 = (__int64 *)v68;
  v67 = 0x1000000000LL;
  v7 = *(_QWORD *)(*v6 + 16);
  if ( v7 )
  {
    v9 = (char *)a2;
    while ( 1 )
    {
      v10 = *(_QWORD *)(v7 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v10 - 30) <= 0xAu )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        return;
    }
    v11 = *(_QWORD *)(v10 + 40);
    v58 = a2 + 56;
    if ( !*(_BYTE *)(a2 + 84) )
      goto LABEL_15;
LABEL_5:
    v12 = *(_QWORD **)(a2 + 64);
    v13 = &v12[*(unsigned int *)(a2 + 76)];
    if ( v12 != v13 )
    {
      while ( v11 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_12;
      }
LABEL_9:
      v14 = (unsigned int)v67;
      v15 = (unsigned int)v67 + 1LL;
      if ( v15 > HIDWORD(v67) )
      {
        sub_C8D5F0((__int64)&v66, v68, v15, 8u, a5, a6);
        v14 = (unsigned int)v67;
      }
      v66[v14] = v11;
      LODWORD(v67) = v67 + 1;
    }
LABEL_12:
    while ( 1 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        break;
      while ( 1 )
      {
        v16 = *(_QWORD *)(v7 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v16 - 30) > 0xAu )
          break;
        v11 = *(_QWORD *)(v16 + 40);
        if ( *(_BYTE *)(a2 + 84) )
          goto LABEL_5;
LABEL_15:
        if ( sub_C8CA60(v58, v11) )
          goto LABEL_9;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          goto LABEL_17;
      }
    }
LABEL_17:
    v17 = v66;
    v55 = &v66[(unsigned int)v67];
    if ( v66 != v55 )
    {
      v59 = v66;
      while ( 1 )
      {
        v18 = *v59;
        if ( (_BYTE)qword_5004B68 )
        {
          v56 = v18 + 48;
          goto LABEL_34;
        }
        v52 = *(__int64 **)(a1 + 208);
        v19 = sub_DCF3A0(v52, v9, 1);
        if ( !sub_D96A50(v19) )
        {
          v38 = sub_DBB9F0((__int64)v52, v19, 0, 0);
          v63 = *(_DWORD *)(v38 + 8);
          if ( v63 > 0x40 )
            sub_C43780((__int64)&v62, (const void **)v38);
          else
            v62 = *(_QWORD *)v38;
          v65 = *(_DWORD *)(v38 + 24);
          if ( v65 > 0x40 )
            sub_C43780((__int64)&v64, (const void **)(v38 + 16));
          else
            v64 = *(_QWORD *)(v38 + 16);
          sub_AB0910((__int64)&v60, (__int64)&v62);
          v39 = v61;
          if ( v61 > 0x40 )
          {
            v57 = qword_5004A88;
            v40 = v39 - (unsigned int)sub_C444A0((__int64)&v60) <= v57;
            if ( v60 )
              j_j___libc_free_0_0(v60);
          }
          else
          {
            v40 = 1;
            if ( v60 )
            {
              _BitScanReverse64(&v41, v60);
              v40 = (unsigned int)qword_5004A88 >= 64 - ((unsigned int)v41 ^ 0x3F);
            }
          }
          if ( v65 > 0x40 && v64 )
            j_j___libc_free_0_0(v64);
          if ( v63 > 0x40 && v62 )
            j_j___libc_free_0_0(v62);
          if ( v40 )
            goto LABEL_42;
        }
        v56 = v18 + 48;
        v20 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v18 + 48 == v20 )
          goto LABEL_33;
        if ( !v20 )
          BUG();
        v21 = v20 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA || (v22 = sub_B46E30(v21)) == 0 )
        {
LABEL_33:
          if ( *(_BYTE *)(a1 + 200) )
          {
            v51 = a1;
            v42 = v18;
            v43 = *(__int64 **)(a1 + 232);
            v48 = v18;
            v53 = *(_QWORD *)(a1 + 216);
            for ( i = v56; ; i = v42 + 48 )
            {
              v45 = *(_QWORD *)(v42 + 56);
              if ( v45 != i )
                break;
LABEL_90:
              if ( v49 == v42 )
              {
                v18 = v48;
                a1 = v51;
                goto LABEL_34;
              }
              v47 = (unsigned int)(*(_DWORD *)(v42 + 44) + 1);
              if ( (unsigned int)v47 >= *(_DWORD *)(v53 + 32) )
                BUG();
              v42 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v53 + 24) + 8 * v47) + 8LL);
            }
            while ( 1 )
            {
              if ( !v45 )
                goto LABEL_106;
              if ( (unsigned __int8)(*(_BYTE *)(v45 - 24) - 34) <= 0x33u )
              {
                v46 = 0x8000000000041LL;
                if ( _bittest64(&v46, (unsigned int)*(unsigned __int8 *)(v45 - 24) - 34) )
                {
                  if ( sub_28E5B80(v45 - 24, v43) )
                    break;
                }
              }
              v45 = *(_QWORD *)(v45 + 8);
              if ( v45 == i )
                goto LABEL_90;
            }
            a1 = v51;
            goto LABEL_42;
          }
LABEL_34:
          v29 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v29 == v56 )
          {
            v31 = 0;
          }
          else
          {
            if ( !v29 )
LABEL_106:
              BUG();
            v30 = *(unsigned __int8 *)(v29 - 24);
            v31 = v29 - 24;
            if ( (unsigned int)(v30 - 30) >= 0xB )
              v31 = 0;
          }
          v62 = v31;
          v32 = *(_BYTE **)(a1 + 184);
          if ( v32 == *(_BYTE **)(a1 + 192) )
          {
            sub_24454E0(a1 + 176, v32, &v62);
          }
          else
          {
            if ( v32 )
            {
              *(_QWORD *)v32 = v31;
              v32 = *(_BYTE **)(a1 + 184);
            }
            *(_QWORD *)(a1 + 184) = v32 + 8;
          }
          goto LABEL_42;
        }
        v50 = v18;
        v23 = v9;
        v24 = 0;
        v25 = v22;
        while ( 1 )
        {
          v26 = sub_B46EC0(v21, v24);
          if ( !v23[84] )
            break;
          v27 = (_QWORD *)*((_QWORD *)v23 + 8);
          v28 = &v27[*((unsigned int *)v23 + 19)];
          if ( v27 == v28 )
            goto LABEL_50;
          while ( v26 != *v27 )
          {
            if ( v28 == ++v27 )
              goto LABEL_50;
          }
LABEL_31:
          if ( v25 == ++v24 )
          {
            v9 = v23;
            v18 = v50;
            goto LABEL_33;
          }
        }
        if ( sub_C8CA60(v58, v26) )
          goto LABEL_31;
LABEL_50:
        v9 = v23;
        v18 = v50;
        v33 = sub_DBA6E0((__int64)v52, (__int64)v9, v50, 0);
        if ( sub_D96A50(v33) )
          goto LABEL_33;
        v34 = sub_DBB9F0((__int64)v52, v33, 0, 0);
        v63 = *(_DWORD *)(v34 + 8);
        if ( v63 > 0x40 )
          sub_C43780((__int64)&v62, (const void **)v34);
        else
          v62 = *(_QWORD *)v34;
        v65 = *(_DWORD *)(v34 + 24);
        if ( v65 > 0x40 )
          sub_C43780((__int64)&v64, (const void **)(v34 + 16));
        else
          v64 = *(_QWORD *)(v34 + 16);
        sub_AB0910((__int64)&v60, (__int64)&v62);
        v35 = v61;
        if ( v61 > 0x40 )
        {
          v54 = qword_5004A88;
          v36 = v54 >= v35 - (unsigned int)sub_C444A0((__int64)&v60);
          if ( v60 )
            j_j___libc_free_0_0(v60);
        }
        else
        {
          v36 = 1;
          if ( v60 )
          {
            _BitScanReverse64(&v37, v60);
            v36 = (unsigned int)qword_5004A88 >= 64 - ((unsigned int)v37 ^ 0x3F);
          }
        }
        if ( v65 > 0x40 && v64 )
          j_j___libc_free_0_0(v64);
        if ( v63 > 0x40 && v62 )
          j_j___libc_free_0_0(v62);
        if ( !v36 )
          goto LABEL_33;
LABEL_42:
        if ( v55 == ++v59 )
        {
          v17 = v66;
          break;
        }
      }
    }
    if ( v17 != (__int64 *)v68 )
      _libc_free((unsigned __int64)v17);
  }
}
