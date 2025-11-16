// Function: sub_1A922B0
// Address: 0x1a922b0
//
void __fastcall sub_1A922B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // r12
  _QWORD *v4; // rax
  __int64 v5; // r14
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rax
  __int64 *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r15
  unsigned __int64 v12; // r13
  __int64 *v13; // r13
  unsigned int v14; // r13d
  bool v15; // r13
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  _BYTE *v18; // rsi
  unsigned __int64 v19; // rdi
  int v20; // r13d
  unsigned __int64 v21; // r14
  unsigned int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 *v25; // r13
  unsigned int v26; // r13d
  unsigned int v27; // r15d
  int v28; // r14d
  bool v29; // r13
  __int64 v30; // r14
  __int64 i; // r13
  __int64 v32; // rbx
  unsigned __int8 v33; // dl
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int8 v37; // dl
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rsi
  unsigned int v41; // ecx
  __int64 *v42; // rdx
  __int64 v43; // r8
  int v44; // eax
  unsigned int v45; // edx
  int v46; // r14d
  unsigned __int64 v47; // rax
  int v48; // edx
  int v49; // r9d
  __int64 v50; // [rsp+8h] [rbp-148h]
  __int64 v51; // [rsp+10h] [rbp-140h]
  __int64 v52; // [rsp+28h] [rbp-128h]
  __int64 v53; // [rsp+30h] [rbp-120h]
  __int64 v54; // [rsp+30h] [rbp-120h]
  int v55; // [rsp+30h] [rbp-120h]
  __int64 v56; // [rsp+38h] [rbp-118h]
  __int64 *v58; // [rsp+50h] [rbp-100h]
  __int64 v59; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v60; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v61; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v62; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v63; // [rsp+78h] [rbp-D8h]
  __int64 v64; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int v65; // [rsp+88h] [rbp-C8h]
  __int64 *v66; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+98h] [rbp-B8h]
  _BYTE v68[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v2 = *(__int64 **)(a2 + 32);
  v56 = *v2;
  v66 = (__int64 *)v68;
  v67 = 0x1000000000LL;
  v3 = *(_QWORD *)(*v2 + 8);
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = sub_1648700(v3);
      if ( (unsigned __int8)(*((_BYTE *)v4 + 16) - 25) <= 9u )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return;
    }
    v52 = a2 + 56;
LABEL_6:
    v5 = v4[5];
    if ( sub_1377F70(a2 + 56, v5) )
    {
      v8 = (unsigned int)v67;
      if ( (unsigned int)v67 >= HIDWORD(v67) )
      {
        sub_16CD150((__int64)&v66, v68, 0, 8, v6, v7);
        v8 = (unsigned int)v67;
      }
      v66[v8] = v5;
      LODWORD(v67) = v67 + 1;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v3 )
        goto LABEL_5;
    }
    else
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          break;
LABEL_5:
        v4 = sub_1648700(v3);
        if ( (unsigned __int8)(*((_BYTE *)v4 + 16) - 25) <= 9u )
          goto LABEL_6;
      }
    }
    v58 = &v66[(unsigned int)v67];
    if ( v58 == v66 )
      goto LABEL_30;
    v9 = v66;
    v10 = a1;
    v50 = a1 + 160;
    while ( 1 )
    {
      v59 = *v9;
      if ( byte_4FB5D00 )
        goto LABEL_37;
      v11 = *(_QWORD *)(v10 + 192);
      v12 = sub_1474260(v11, a2);
      if ( v12 != sub_1456E90(v11) )
      {
        v13 = sub_1477920(v11, v12, 0);
        v63 = *((_DWORD *)v13 + 2);
        if ( v63 > 0x40 )
          sub_16A4FD0((__int64)&v62, (const void **)v13);
        else
          v62 = *v13;
        v65 = *((_DWORD *)v13 + 6);
        if ( v65 > 0x40 )
          sub_16A4FD0((__int64)&v64, (const void **)v13 + 2);
        else
          v64 = v13[2];
        sub_158A9F0((__int64)&v60, (__int64)&v62);
        v14 = v61;
        if ( v61 > 0x40 )
        {
          v55 = dword_4FB5C20;
          v44 = sub_16A57B0((__int64)&v60);
          v45 = v55;
          v46 = v44;
          if ( v60 )
          {
            j_j___libc_free_0_0(v60);
            v45 = v55;
          }
          v15 = v45 >= v14 - v46;
        }
        else
        {
          v15 = 1;
          if ( v60 )
          {
            _BitScanReverse64(&v16, v60);
            v15 = 64 - ((unsigned int)v16 ^ 0x3F) <= dword_4FB5C20;
          }
        }
        if ( v65 > 0x40 && v64 )
          j_j___libc_free_0_0(v64);
        if ( v63 > 0x40 && v62 )
          j_j___libc_free_0_0(v62);
        if ( v15 )
          goto LABEL_28;
      }
      v19 = sub_157EBA0(v59);
      if ( !v19 )
        goto LABEL_36;
      v20 = sub_15F4D60(v19);
      v21 = sub_157EBA0(v59);
      if ( !v20 )
        goto LABEL_36;
      v53 = v10;
      v22 = 0;
      while ( 1 )
      {
        v23 = sub_15F4DF0(v21, v22);
        if ( !sub_1377F70(v52, v23) )
          break;
        if ( v20 == ++v22 )
        {
          v10 = v53;
          goto LABEL_36;
        }
      }
      v10 = v53;
      v24 = sub_1474160(v11, a2, v59);
      if ( v24 == sub_1456E90(v11) )
      {
LABEL_36:
        if ( !*(_BYTE *)(v10 + 184) )
          goto LABEL_37;
LABEL_62:
        v51 = v10;
        v30 = *(_QWORD *)(v10 + 216);
        v54 = *(_QWORD *)(v10 + 200);
        for ( i = v59; ; i = **(_QWORD **)(v42[1] + 8) )
        {
          v32 = *(_QWORD *)(i + 48);
          if ( v32 != i + 40 )
            break;
LABEL_81:
          if ( v56 == i )
          {
            v10 = v51;
            goto LABEL_37;
          }
          v39 = *(unsigned int *)(v54 + 48);
          if ( !(_DWORD)v39 )
            goto LABEL_102;
          v40 = *(_QWORD *)(v54 + 32);
          v41 = (v39 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v42 = (__int64 *)(v40 + 16LL * v41);
          v43 = *v42;
          if ( i != *v42 )
          {
            v48 = 1;
            while ( v43 != -8 )
            {
              v49 = v48 + 1;
              v41 = (v39 - 1) & (v48 + v41);
              v42 = (__int64 *)(v40 + 16LL * v41);
              v43 = *v42;
              if ( i == *v42 )
                goto LABEL_84;
              v48 = v49;
            }
LABEL_102:
            BUG();
          }
LABEL_84:
          if ( v42 == (__int64 *)(v40 + 16 * v39) )
            goto LABEL_102;
        }
        while ( 2 )
        {
          if ( !v32 )
          {
            v62 = 0;
            BUG();
          }
          v33 = *(_BYTE *)(v32 - 8);
          v34 = v32 - 24;
          if ( v33 > 0x17u )
          {
            if ( v33 == 78 )
            {
              v35 = v34 & 0xFFFFFFFFFFFFFFF8LL;
              v36 = (v32 - 24) | 4;
              goto LABEL_71;
            }
            if ( v33 == 29 )
            {
              v35 = v34 & 0xFFFFFFFFFFFFFFF8LL;
              v36 = (v32 - 24) & 0xFFFFFFFFFFFFFFFBLL;
LABEL_71:
              v62 = v36;
              if ( v35 )
              {
                v37 = *(_BYTE *)(v35 + 16);
                v38 = 0;
                if ( v37 > 0x17u )
                {
                  if ( v37 == 78 )
                  {
                    v38 = v35 | 4;
                  }
                  else
                  {
                    v38 = 0;
                    if ( v37 == 29 )
                      v38 = v35;
                  }
                }
                if ( !(unsigned __int8)sub_1AEC650(v38, v30) && (unsigned __int8)sub_1A91760(&v62) )
                {
                  v10 = v51;
                  goto LABEL_28;
                }
              }
            }
          }
          v32 = *(_QWORD *)(v32 + 8);
          if ( i + 40 == v32 )
            goto LABEL_81;
          continue;
        }
      }
      v25 = sub_1477920(v11, v24, 0);
      v63 = *((_DWORD *)v25 + 2);
      if ( v63 > 0x40 )
      {
        sub_16A4FD0((__int64)&v62, (const void **)v25);
        v65 = *((_DWORD *)v25 + 6);
        if ( v65 <= 0x40 )
        {
LABEL_49:
          v64 = v25[2];
          goto LABEL_50;
        }
      }
      else
      {
        v62 = *v25;
        v65 = *((_DWORD *)v25 + 6);
        if ( v65 <= 0x40 )
          goto LABEL_49;
      }
      sub_16A4FD0((__int64)&v64, (const void **)v25 + 2);
LABEL_50:
      sub_158A9F0((__int64)&v60, (__int64)&v62);
      v26 = v61;
      v27 = dword_4FB5C20;
      if ( v61 <= 0x40 )
      {
        v29 = 1;
        if ( v60 )
        {
          _BitScanReverse64(&v47, v60);
          v29 = dword_4FB5C20 >= 64 - ((unsigned int)v47 ^ 0x3F);
        }
      }
      else
      {
        v28 = sub_16A57B0((__int64)&v60);
        if ( v60 )
          j_j___libc_free_0_0(v60);
        v29 = v27 >= v26 - v28;
      }
      if ( v65 > 0x40 && v64 )
        j_j___libc_free_0_0(v64);
      if ( v63 > 0x40 && v62 )
        j_j___libc_free_0_0(v62);
      if ( v29 )
        goto LABEL_28;
      if ( *(_BYTE *)(v53 + 184) )
        goto LABEL_62;
LABEL_37:
      v17 = sub_157EBA0(v59);
      v18 = *(_BYTE **)(v10 + 168);
      v62 = v17;
      if ( v18 == *(_BYTE **)(v10 + 176) )
      {
        sub_1A92120(v50, v18, &v62);
      }
      else
      {
        if ( v18 )
        {
          *(_QWORD *)v18 = v17;
          v18 = *(_BYTE **)(v10 + 168);
        }
        *(_QWORD *)(v10 + 168) = v18 + 8;
      }
LABEL_28:
      if ( v58 == ++v9 )
      {
        v58 = v66;
LABEL_30:
        if ( v58 != (__int64 *)v68 )
          _libc_free((unsigned __int64)v58);
        return;
      }
    }
  }
}
