// Function: sub_1524F60
// Address: 0x1524f60
//
void __fastcall sub_1524F60(_QWORD *a1)
{
  _QWORD *v1; // r12
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // rdx
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rax
  int v9; // edx
  int v10; // r14d
  __int64 v11; // rax
  int v12; // ebx
  _DWORD *v13; // r15
  __int64 v14; // rbx
  int v15; // edx
  _DWORD *v16; // rdi
  unsigned int v17; // r14d
  unsigned int v18; // ebx
  int v19; // ecx
  unsigned int v20; // eax
  unsigned int v21; // r14d
  unsigned int v22; // ecx
  int v23; // r14d
  __int64 v24; // r12
  __int64 v25; // rdx
  unsigned int v26; // ecx
  int v27; // edx
  unsigned int v28; // r14d
  __int64 v29; // rbx
  __int64 v30; // r14
  unsigned int v31; // esi
  _QWORD *v32; // [rsp+8h] [rbp-158h]
  _QWORD *v33; // [rsp+10h] [rbp-150h]
  unsigned int v34; // [rsp+18h] [rbp-148h]
  _BYTE *v35; // [rsp+20h] [rbp-140h] BYREF
  __int64 v36; // [rsp+28h] [rbp-138h]
  _BYTE v37[304]; // [rsp+30h] [rbp-130h] BYREF

  v1 = (_QWORD *)a1[26];
  v35 = v37;
  v36 = 0x4000000000LL;
  v32 = (_QWORD *)a1[27];
  if ( v1 != v32 )
  {
    while ( 1 )
    {
      v3 = *v1;
      v4 = sub_1580C70(*v1);
      v6 = sub_15240F0((__int64)a1, v4, v5);
      v7 = (unsigned int)v36;
      if ( (unsigned int)v36 >= HIDWORD(v36) )
      {
        sub_16CD150(&v35, v37, 0, 4);
        v7 = (unsigned int)v36;
      }
      *(_DWORD *)&v35[4 * v7] = v6;
      LODWORD(v36) = v36 + 1;
      sub_1580C70(v3);
      v8 = (unsigned int)v36;
      v10 = v9;
      if ( (unsigned int)v36 >= HIDWORD(v36) )
      {
        sub_16CD150(&v35, v37, 0, 4);
        v8 = (unsigned int)v36;
      }
      *(_DWORD *)&v35[4 * v8] = v10;
      v11 = (unsigned int)(v36 + 1);
      LODWORD(v36) = v11;
      v12 = *(_DWORD *)(v3 + 8) + 1;
      if ( (unsigned int)v11 >= HIDWORD(v36) )
      {
        sub_16CD150(&v35, v37, 0, 4);
        v11 = (unsigned int)v36;
      }
      *(_DWORD *)&v35[4 * v11] = v12;
      v13 = (_DWORD *)*a1;
      v14 = (unsigned int)v36;
      v15 = *(_DWORD *)(*a1 + 16LL);
      v16 = (_DWORD *)*a1;
      LODWORD(v36) = v36 + 1;
      v17 = v36;
      sub_1524D80(v16, 3u, v15);
      sub_1524E40(v13, 0xCu, 6);
      if ( v17 > 0x1F )
        break;
      sub_1524D80(v13, v17, 6);
      if ( v17 )
        goto LABEL_18;
LABEL_20:
      LODWORD(v36) = 0;
      if ( v32 == ++v1 )
      {
        if ( v35 != v37 )
          _libc_free((unsigned __int64)v35);
        return;
      }
    }
    v34 = v14;
    v18 = v17;
    v33 = v1;
    do
    {
      while ( 1 )
      {
        v19 = v13[2];
        v20 = v18 & 0x1F | 0x20;
        v21 = v20 << v19;
        v22 = v19 + 6;
        v23 = v13[3] | v21;
        v13[3] = v23;
        if ( v22 > 0x1F )
          break;
        v18 >>= 5;
        v13[2] = v22;
        if ( v18 <= 0x1F )
          goto LABEL_17;
      }
      v24 = *(_QWORD *)v13;
      v25 = *(unsigned int *)(*(_QWORD *)v13 + 8LL);
      if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)v13 + 12LL) - v25 <= 3 )
      {
        sub_16CD150(*(_QWORD *)v13, v24 + 16, v25 + 4, 1);
        v25 = *(unsigned int *)(v24 + 8);
        v20 = v18 & 0x1F | 0x20;
      }
      *(_DWORD *)(*(_QWORD *)v24 + v25) = v23;
      v26 = 0;
      *(_DWORD *)(v24 + 8) += 4;
      v27 = v13[2];
      if ( v27 )
        v26 = v20 >> (32 - v27);
      v18 >>= 5;
      v13[3] = v26;
      v13[2] = ((_BYTE)v27 + 6) & 0x1F;
    }
    while ( v18 > 0x1F );
LABEL_17:
    v28 = v18;
    v14 = v34;
    v1 = v33;
    sub_1524D80(v13, v28, 6);
LABEL_18:
    v29 = 4 * v14 + 4;
    v30 = 0;
    do
    {
      v31 = *(_DWORD *)&v35[v30];
      v30 += 4;
      sub_1524E40(v13, v31, 6);
    }
    while ( v29 != v30 );
    goto LABEL_20;
  }
}
