// Function: sub_2A28870
// Address: 0x2a28870
//
__int64 __fastcall sub_2A28870(
        __int64 a1,
        __int64 *a2,
        const void *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v12; // rax
  _QWORD *i; // rdx
  signed __int64 v14; // r12
  void *v15; // rdi
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rax
  int v20; // [rsp+8h] [rbp-68h]

  *(_QWORD *)a1 = a5;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 40) = 128;
  v12 = (_QWORD *)sub_C7D670(0x2000, 8);
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) = v12;
  for ( i = &v12[8 * (unsigned __int64)*(unsigned int *)(a1 + 40)]; i != v12; v12 += 8 )
  {
    if ( v12 )
    {
      v12[2] = 0;
      v12[3] = -4096;
      *v12 = &unk_49DD7B0;
      v12[1] = 2;
      v12[4] = 0;
    }
  }
  v14 = 16 * a4;
  v15 = (void *)(a1 + 112);
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x400000000LL;
  v16 = v14 >> 4;
  if ( (unsigned __int64)v14 > 0x40 )
  {
    sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), v14 >> 4, 0x10u, v16, a1 + 96);
    v16 = v14 >> 4;
    v15 = (void *)(*(_QWORD *)(a1 + 96) + 16LL * *(unsigned int *)(a1 + 104));
    goto LABEL_9;
  }
  if ( v14 )
  {
LABEL_9:
    v20 = v16;
    memcpy(v15, a3, v14);
    LODWORD(v14) = *(_DWORD *)(a1 + 104);
    LODWORD(v16) = v20;
  }
  v17 = *a2;
  *(_DWORD *)(a1 + 104) = v16 + v14;
  v18 = sub_D9B120(v17);
  *(_QWORD *)(a1 + 280) = a2;
  *(_QWORD *)(a1 + 176) = v18;
  *(_QWORD *)(a1 + 288) = a6;
  *(_QWORD *)(a1 + 296) = a7;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_DWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_DWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_DWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 304) = a8;
  return a8;
}
