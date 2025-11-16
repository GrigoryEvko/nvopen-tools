// Function: sub_2B7A390
// Address: 0x2b7a390
//
_BYTE *__fastcall sub_2B7A390(__int64 a1, __int64 a2, __int64 a3, void *a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int **v9; // r12
  unsigned __int64 v10; // rcx
  unsigned int v11; // edx
  unsigned int v12; // eax
  char v13; // al
  unsigned int **v14; // r15
  _BYTE *v15; // r12
  __int64 v16; // rbx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rsi
  char v21; // al
  _QWORD *v22; // rax
  unsigned int *v23; // rbx
  __int64 v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // rax
  __int64 v28; // r15
  _BYTE *v31; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v32[4]; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v33; // [rsp+40h] [rbp-90h]
  __m128i v34; // [rsp+50h] [rbp-80h] BYREF
  __int64 v35; // [rsp+60h] [rbp-70h]
  __int64 v36; // [rsp+68h] [rbp-68h]
  __int64 v37; // [rsp+70h] [rbp-60h]
  __int64 v38; // [rsp+78h] [rbp-58h]
  __int64 v39; // [rsp+80h] [rbp-50h]
  __int64 v40; // [rsp+88h] [rbp-48h]
  __int16 v41; // [rsp+90h] [rbp-40h]

  v5 = a2;
  v6 = a3;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(_QWORD *)(a3 + 8);
  if ( v7 != v8 )
  {
    v9 = *(unsigned int ***)a1;
    v10 = *(_QWORD *)(a1 + 24);
    v11 = *(_DWORD *)(*(_QWORD *)(v8 + 24) + 8LL);
    v12 = *(_DWORD *)(*(_QWORD *)(v7 + 24) + 8LL);
    v33 = 257;
    v34 = (__m128i)v10;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 257;
    if ( v11 >> 8 >= v12 >> 8 )
    {
      v21 = sub_9AC470(a2, &v34, 0);
      v5 = sub_921630(v9, a2, *(_QWORD *)(v6 + 8), v21 ^ 1u, (__int64)v32);
    }
    else
    {
      v13 = sub_9AC470(v6, &v34, 0);
      v6 = sub_921630(v9, v6, *(_QWORD *)(a2 + 8), v13 ^ 1u, (__int64)v32);
    }
  }
  v14 = *(unsigned int ***)a1;
  v33 = 257;
  v15 = (_BYTE *)(*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, void *, __int64))(*(_QWORD *)v14[10]
                                                                                              + 112LL))(
                   v14[10],
                   v5,
                   v6,
                   a4,
                   a5);
  if ( !v15 )
  {
    LOWORD(v37) = 257;
    v22 = sub_BD2C40(112, unk_3F1FE60);
    v15 = v22;
    if ( v22 )
      sub_B4E9E0((__int64)v22, v5, v6, a4, a5, (__int64)&v34, 0, 0);
    (*(void (__fastcall **)(unsigned int *, _BYTE *, __int64 *, unsigned int *, unsigned int *))(*(_QWORD *)v14[11]
                                                                                               + 16LL))(
      v14[11],
      v15,
      v32,
      v14[7],
      v14[8]);
    v23 = *v14;
    v24 = (__int64)&(*v14)[4 * *((unsigned int *)v14 + 2)];
    if ( *v14 != (unsigned int *)v24 )
    {
      do
      {
        v25 = *((_QWORD *)v23 + 1);
        v26 = *v23;
        v23 += 4;
        sub_B99FD0((__int64)v15, v26, v25);
      }
      while ( (unsigned int *)v24 != v23 );
    }
  }
  if ( *v15 > 0x1Cu )
  {
    v16 = *(_QWORD *)(a1 + 8);
    v31 = v15;
    sub_2400480((__int64)&v34, v16, (__int64 *)&v31);
    if ( (_BYTE)v37 )
    {
      v27 = *(unsigned int *)(v16 + 40);
      v28 = (__int64)v31;
      if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 44) )
      {
        sub_C8D5F0(v16 + 32, (const void *)(v16 + 48), v27 + 1, 8u, v17, v18);
        v27 = *(unsigned int *)(v16 + 40);
      }
      *(_QWORD *)(*(_QWORD *)(v16 + 32) + 8 * v27) = v28;
      ++*(_DWORD *)(v16 + 40);
    }
    v19 = *(_QWORD *)(a1 + 16);
    v32[0] = *((_QWORD *)v31 + 5);
    sub_29B09C0((__int64)&v34, v19, v32);
  }
  return v15;
}
