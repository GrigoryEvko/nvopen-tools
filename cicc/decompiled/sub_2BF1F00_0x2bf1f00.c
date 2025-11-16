// Function: sub_2BF1F00
// Address: 0x2bf1f00
//
void __fastcall sub_2BF1F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  __int64 j; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 *v10; // r13
  __int64 v11; // r12
  __int64 *v12; // r12
  __int64 v13; // rdi
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // r10
  __int64 *v17; // r10
  __int64 v18; // r15
  _QWORD *v19; // rdi
  __int64 v20; // rsi
  _QWORD *v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // r10
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  _QWORD *v27; // rbx
  _QWORD *k; // r12
  __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  __int64 i; // [rsp+10h] [rbp-C0h]
  _QWORD *v38; // [rsp+18h] [rbp-B8h]
  __int64 v39; // [rsp+28h] [rbp-A8h]
  _QWORD *v40; // [rsp+48h] [rbp-88h]
  __int64 v41; // [rsp+58h] [rbp-78h] BYREF
  _BYTE v42[16]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v43; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-58h]
  unsigned int v45; // [rsp+7Ch] [rbp-54h]
  _BYTE v46[80]; // [rsp+80h] [rbp-50h] BYREF

  v6 = v42;
  sub_2BF0340((__int64)v42, 0, 0, 0, a5, a6);
  v39 = *(_QWORD *)(a1 + 592);
  for ( i = v39 + 8LL * *(unsigned int *)(a1 + 600); i != v39; v39 += 8 )
  {
    v38 = *(_QWORD **)v39;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)v39 + 8LL) - 1 <= 1 )
    {
      for ( j = v38[15]; v38 + 14 != (_QWORD *)j; j = *(_QWORD *)(j + 8) )
      {
        if ( !j )
          BUG();
        v8 = *(_QWORD *)(j - 8);
        v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( (v8 & 4) != 0 )
          {
            v10 = *(__int64 **)v9;
            v11 = *(unsigned int *)(v9 + 8);
          }
          else
          {
            v10 = (__int64 *)(j - 8);
            v11 = 1;
          }
          v12 = &v10[v11];
          while ( v12 != v10 )
          {
            v13 = *v10++;
            sub_2BF1250(v13, (__int64)v6);
          }
        }
        v14 = *(unsigned int *)(j + 32);
        if ( (_DWORD)v14 )
        {
          v40 = v6;
          v15 = 0;
          do
          {
            v16 = *(_QWORD *)(j + 24);
            v41 = j + 16;
            v17 = (__int64 *)(v15 + v16);
            v18 = *v17;
            v19 = *(_QWORD **)(*v17 + 16);
            v20 = (__int64)&v19[*(unsigned int *)(*v17 + 24)];
            v21 = sub_2BEF3B0(v19, v20, &v41);
            if ( (_QWORD *)v20 != v21 )
            {
              if ( (_QWORD *)v20 != v21 + 1 )
              {
                memmove(v21, v21 + 1, v20 - (_QWORD)(v21 + 1));
                LODWORD(v23) = *(_DWORD *)(v18 + 24);
              }
              v23 = (unsigned int)(v23 - 1);
              *(_DWORD *)(v18 + 24) = v23;
              v24 = (_QWORD *)(v15 + *(_QWORD *)(j + 24));
            }
            *v24 = v40;
            v25 = v44;
            v26 = v44 + 1LL;
            if ( v26 > v45 )
            {
              sub_C8D5F0((__int64)&v43, v46, v26, 8u, v22, v23);
              v25 = v44;
            }
            v15 += 8;
            *(_QWORD *)(v43 + 8 * v25) = j + 16;
            ++v44;
          }
          while ( v15 != 8 * v14 );
          v6 = v40;
        }
      }
    }
    (*(void (__fastcall **)(_QWORD *))(*v38 + 8LL))(v38);
  }
  v27 = *(_QWORD **)(a1 + 416);
  for ( k = &v27[*(unsigned int *)(a1 + 424)]; k != v27; ++v27 )
  {
    if ( *v27 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v27 + 8LL))(*v27);
  }
  v29 = *(_QWORD *)(a1 + 208);
  if ( v29 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
  sub_2BF1E70(v6);
  v30 = *(_QWORD *)(a1 + 592);
  if ( v30 != a1 + 608 )
    _libc_free(v30);
  sub_C7D6A0(*(_QWORD *)(a1 + 568), 16LL * *(unsigned int *)(a1 + 584), 8);
  v31 = *(_QWORD *)(a1 + 416);
  if ( v31 != a1 + 432 )
    _libc_free(v31);
  sub_C7D6A0(*(_QWORD *)(a1 + 392), 16LL * *(unsigned int *)(a1 + 408), 8);
  sub_2BF1E70((_QWORD *)(a1 + 328));
  sub_2BF1E70((_QWORD *)(a1 + 272));
  sub_2BF1E70((_QWORD *)(a1 + 216));
  v32 = *(_QWORD *)(a1 + 168);
  if ( v32 != a1 + 184 )
    j_j___libc_free_0(v32);
  v33 = *(_QWORD *)(a1 + 144);
  if ( v33 != a1 + 160 )
    _libc_free(v33);
  sub_C7D6A0(*(_QWORD *)(a1 + 120), 4LL * *(unsigned int *)(a1 + 136), 4);
  v34 = *(_QWORD *)(a1 + 80);
  if ( v34 != a1 + 96 )
    _libc_free(v34);
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 8LL * *(unsigned int *)(a1 + 72), 4);
  v35 = *(_QWORD *)(a1 + 16);
  if ( v35 != a1 + 32 )
    _libc_free(v35);
}
