// Function: sub_350A960
// Address: 0x350a960
//
void __fastcall sub_350A960(
        __int64 a1,
        __m128i a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __int64 a10,
        _QWORD *a11,
        __int64 a12,
        __int64 a13,
        __int64 a14)
{
  __int64 v15; // rdx
  int v16; // eax
  int v18; // ebx
  __int64 v19; // r12
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // r8
  int v23; // r10d
  unsigned __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // r15
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rax
  unsigned __int64 v30; // r11
  __int64 v31; // r12
  unsigned __int64 v32; // rax
  _QWORD *v33; // rcx
  _QWORD *v34; // rdi
  unsigned __int64 v35; // [rsp+8h] [rbp-48h]
  int v36; // [rsp+10h] [rbp-40h]
  int v37; // [rsp+14h] [rbp-3Ch]
  _QWORD *v38; // [rsp+18h] [rbp-38h]
  __int64 v39; // [rsp+18h] [rbp-38h]

  v15 = *(_QWORD *)(a1 + 16);
  v16 = *(_DWORD *)(a1 + 64);
  v37 = *(_DWORD *)(v15 + 8) - v16;
  if ( v37 )
  {
    v18 = 0;
    while ( 1 )
    {
      v22 = *(_QWORD *)(a1 + 32);
      v23 = *(_DWORD *)(*(_QWORD *)v15 + 4LL * (unsigned int)(v18 + v16));
      v24 = *(unsigned int *)(v22 + 160);
      v25 = v23 & 0x7FFFFFFF;
      v26 = 8LL * (v23 & 0x7FFFFFFF);
      if ( (v23 & 0x7FFFFFFFu) >= (unsigned int)v24 )
        break;
      v19 = *(_QWORD *)(*(_QWORD *)(v22 + 152) + 8LL * v25);
      if ( !v19 )
        break;
LABEL_4:
      ++v18;
      sub_2EBE5D0(*(_QWORD **)(a1 + 24), *(_DWORD *)(v19 + 112));
      sub_34C9770(a11, v19, a2, a3, a4, a5, v20, v21, a8, a9);
      if ( v18 == v37 )
        return;
      v15 = *(_QWORD *)(a1 + 16);
      v16 = *(_DWORD *)(a1 + 64);
    }
    v27 = v25 + 1;
    if ( (unsigned int)v24 < v27 )
    {
      v30 = v27;
      if ( v27 != v24 )
      {
        if ( v27 >= v24 )
        {
          v31 = *(_QWORD *)(v22 + 168);
          v32 = v27 - v24;
          if ( v30 > *(unsigned int *)(v22 + 164) )
          {
            v35 = v32;
            v36 = v23;
            v39 = *(_QWORD *)(a1 + 32);
            sub_C8D5F0(v22 + 152, (const void *)(v22 + 168), v30, 8u, v22, a14);
            v22 = v39;
            v32 = v35;
            v23 = v36;
            v24 = *(unsigned int *)(v39 + 160);
          }
          v28 = *(_QWORD *)(v22 + 152);
          v33 = (_QWORD *)(v28 + 8 * v24);
          v34 = &v33[v32];
          if ( v33 != v34 )
          {
            do
              *v33++ = v31;
            while ( v34 != v33 );
            LODWORD(v24) = *(_DWORD *)(v22 + 160);
            v28 = *(_QWORD *)(v22 + 152);
          }
          *(_DWORD *)(v22 + 160) = v32 + v24;
          goto LABEL_9;
        }
        *(_DWORD *)(v22 + 160) = v27;
      }
    }
    v28 = *(_QWORD *)(v22 + 152);
LABEL_9:
    v38 = (_QWORD *)v22;
    v29 = sub_2E10F30(v23);
    *(_QWORD *)(v28 + v26) = v29;
    v19 = v29;
    sub_2E11E80(v38, v29);
    goto LABEL_4;
  }
}
