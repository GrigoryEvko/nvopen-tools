// Function: sub_216F7F0
// Address: 0x216f7f0
//
void __fastcall sub_216F7F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  int v5; // r12d
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  void (__fastcall *v11)(__int64, _QWORD, _QWORD); // r15
  __int64 v12; // rax
  _WORD *v13; // r15
  _QWORD *v14; // r12
  unsigned __int64 v15; // r14
  size_t v16; // rax
  char *v17; // rsi
  _WORD *v18; // r8
  _QWORD *v19; // r12
  unsigned __int64 v20; // r14
  size_t v21; // rax
  size_t v22; // r15
  _QWORD *v23; // rdx
  _QWORD *v24; // rdi
  void *v25; // rdx
  _WORD *v26; // rdx
  __int64 v27; // rdi
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  _WORD *dest; // [rsp+10h] [rbp-2B0h]
  char *src; // [rsp+18h] [rbp-2A8h]
  size_t srcb; // [rsp+18h] [rbp-2A8h]
  char *srca; // [rsp+18h] [rbp-2A8h]
  _QWORD v37[2]; // [rsp+20h] [rbp-2A0h] BYREF
  _QWORD *v38; // [rsp+30h] [rbp-290h] BYREF
  __int16 v39; // [rsp+40h] [rbp-280h]
  _QWORD v40[2]; // [rsp+50h] [rbp-270h] BYREF
  __int64 v41; // [rsp+60h] [rbp-260h]
  _WORD *v42; // [rsp+68h] [rbp-258h]
  int v43; // [rsp+70h] [rbp-250h]
  unsigned __int64 *v44; // [rsp+78h] [rbp-248h]
  unsigned __int64 v45[2]; // [rsp+80h] [rbp-240h] BYREF
  _BYTE v46[560]; // [rsp+90h] [rbp-230h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 307LL) )
  {
    v46[1] = 1;
    v45[0] = (unsigned __int64)"llvm.ident";
    v46[0] = 3;
    v3 = sub_1632310(a2, (__int64)v45);
    v4 = v3;
    if ( v3 )
    {
      v5 = sub_161F520(v3);
      if ( v5 )
      {
        v6 = 0;
        while ( 1 )
        {
          v7 = sub_161F530(v4, v6);
          v8 = sub_161E970(*(_QWORD *)(v7 - 8LL * *(unsigned int *)(v7 + 8)));
          if ( v9 == 10 && *(_QWORD *)v8 == 0x6564692E6363766ELL && *(_WORD *)(v8 + 8) == 29806 )
            break;
          if ( v5 == ++v6 )
            return;
        }
        v45[0] = (unsigned __int64)v46;
        v45[1] = 0x20000000000LL;
        v44 = v45;
        v43 = 1;
        v40[0] = &unk_49EFC48;
        v42 = 0;
        v41 = 0;
        v40[1] = 0;
        sub_16E7A40((__int64)v40, 0, 0, 0);
        v10 = *(_QWORD *)(a1 + 256);
        v11 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v10 + 160LL);
        v12 = sub_396DD80(a1);
        v11(v10, *(_QWORD *)(v12 + 680), 0);
        if ( (unsigned __int64)(v41 - (_QWORD)v42) <= 1 )
        {
          sub_16E7EE0((__int64)v40, "\t\"", 2u);
          v13 = v42;
        }
        else
        {
          *v42 = 8713;
          v13 = ++v42;
        }
        v14 = v40;
        v15 = v41 - (_QWORD)v13;
        if ( off_4CD49A0 )
        {
          src = (char *)off_4CD49A0;
          v16 = strlen((const char *)off_4CD49A0);
          v17 = src;
          if ( v15 < v16 )
          {
            v30 = sub_16E7EE0((__int64)v40, src, v16);
            v13 = *(_WORD **)(v30 + 24);
            v14 = (_QWORD *)v30;
            v15 = *(_QWORD *)(v30 + 16) - (_QWORD)v13;
          }
          else if ( v16 )
          {
            srcb = v16;
            memcpy(v13, v17, v16);
            v42 = (_WORD *)((char *)v42 + srcb);
            v13 = v42;
            v15 = v41 - (_QWORD)v42;
          }
        }
        if ( v15 <= 1 )
        {
          sub_16E7EE0((__int64)v14, "; ", 2u);
        }
        else
        {
          *v13 = 8251;
          v14[3] += 2LL;
        }
        v18 = v42;
        v19 = v40;
        v20 = v41 - (_QWORD)v42;
        if ( off_4CD4998 )
        {
          dest = v42;
          srca = (char *)off_4CD4998;
          v21 = strlen((const char *)off_4CD4998);
          v18 = dest;
          v22 = v21;
          if ( v20 < v21 )
          {
            v29 = sub_16E7EE0((__int64)v40, srca, v21);
            v18 = *(_WORD **)(v29 + 24);
            v19 = (_QWORD *)v29;
            v20 = *(_QWORD *)(v29 + 16) - (_QWORD)v18;
          }
          else if ( v21 )
          {
            memcpy(dest, srca, v21);
            v18 = (_WORD *)((char *)v42 + v22);
            v42 = v18;
            v20 = v41 - (_QWORD)v18;
          }
        }
        if ( v20 <= 1 )
        {
          sub_16E7EE0((__int64)v19, "; ", 2u);
        }
        else
        {
          *v18 = 8251;
          v19[3] += 2LL;
        }
        v23 = v42;
        if ( (unsigned __int64)(v41 - (_QWORD)v42) <= 8 )
        {
          v32 = sub_16E7EE0((__int64)v40, "Based on ", 9u);
          v25 = *(void **)(v32 + 24);
          v24 = (_QWORD *)v32;
        }
        else
        {
          *((_BYTE *)v42 + 8) = 32;
          v24 = v40;
          *v23 = 0x6E6F206465736142LL;
          v25 = (char *)v42 + 9;
          v42 = (_WORD *)((char *)v42 + 9);
        }
        if ( v24[2] - (_QWORD)v25 <= 9u )
        {
          v31 = sub_16E7EE0((__int64)v24, "NVVM 7.0.1", 0xAu);
          v26 = *(_WORD **)(v31 + 24);
          v24 = (_QWORD *)v31;
        }
        else
        {
          qmemcpy(v25, "NVVM 7.0.1", 10);
          v26 = (_WORD *)(v24[3] + 10LL);
          v24[3] = v26;
        }
        if ( v24[2] - (_QWORD)v26 <= 1u )
        {
          sub_16E7EE0((__int64)v24, "\"\n", 2u);
        }
        else
        {
          *v26 = 2594;
          v24[3] += 2LL;
        }
        v27 = *(_QWORD *)(a1 + 256);
        v28 = *v44;
        v37[1] = *((unsigned int *)v44 + 2);
        v39 = 261;
        v37[0] = v28;
        v38 = v37;
        sub_38DD5A0(v27, &v38);
        v40[0] = &unk_49EFD28;
        sub_16E7960((__int64)v40);
        if ( (_BYTE *)v45[0] != v46 )
          _libc_free(v45[0]);
      }
    }
  }
}
