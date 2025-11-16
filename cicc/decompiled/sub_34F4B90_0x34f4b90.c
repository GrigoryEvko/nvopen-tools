// Function: sub_34F4B90
// Address: 0x34f4b90
//
__int64 __fastcall sub_34F4B90(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(void); // rax
  __int64 v5; // rax
  __int64 (*v6)(); // rcx
  __int64 v7; // rdx
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // r15
  unsigned int v13; // r13d
  int v14; // eax
  __int64 *v15; // rbx
  __int64 *v16; // r15
  __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  unsigned __int64 v23; // r12
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v27; // rax

  v3 = a2[2];
  *(_QWORD *)(a1 + 216) = v3;
  v4 = *(__int64 (**)(void))(*(_QWORD *)v3 + 488LL);
  if ( v4 != sub_30594C0 )
  {
    v13 = v4();
    if ( !(_BYTE)v13 )
      return v13;
    v3 = *(_QWORD *)(a1 + 216);
  }
  v5 = a2[4];
  *(_QWORD *)(a1 + 208) = v5;
  v6 = *(__int64 (**)())(*(_QWORD *)v3 + 128LL);
  v7 = 0;
  if ( v6 != sub_2DAC790 )
  {
    v7 = ((__int64 (__fastcall *)(__int64, _QWORD *, _QWORD))v6)(v3, a2, 0);
    v5 = *(_QWORD *)(a1 + 208);
  }
  *(_QWORD *)(a1 + 200) = v7;
  v8 = 0;
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)v5 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)v5 + 16LL));
  v10 = *(_QWORD *)(a1 + 208);
  *(_QWORD *)(a1 + 224) = v9;
  v11 = v9;
  if ( *(_BYTE *)(v10 + 48) )
  {
    v27 = sub_22077B0(0xF8u);
    v8 = (__int64 *)v27;
    if ( v27 )
      sub_2DAE000(v27, v10, v11);
    sub_2DAF240(v8);
  }
  v12 = a2[41];
  v13 = 0;
  if ( (_QWORD *)v12 != a2 + 40 )
  {
    do
    {
      v14 = sub_34F3C10(a1, (__int64)a2, v12, (__int64)v8);
      v12 = *(_QWORD *)(v12 + 8);
      v13 |= v14;
    }
    while ( a2 + 40 != (_QWORD *)v12 );
  }
  v15 = *(__int64 **)(a1 + 328);
  v16 = &v15[*(unsigned int *)(a1 + 336)];
  while ( v16 != v15 )
  {
    v17 = *v15++;
    sub_2E88E20(v17);
  }
  *(_DWORD *)(a1 + 336) = 0;
  v18 = *(_QWORD *)(a1 + 296);
  *(_DWORD *)(a1 + 240) = 0;
  sub_34F3890(v18);
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = a1 + 288;
  *(_QWORD *)(a1 + 312) = a1 + 288;
  *(_QWORD *)(a1 + 320) = 0;
  if ( v8 )
  {
    v19 = v8[22];
    if ( (__int64 *)v19 != v8 + 24 )
      _libc_free(v19);
    v20 = v8[13];
    if ( (__int64 *)v20 != v8 + 15 )
      _libc_free(v20);
    v21 = v8[3];
    if ( v21 )
    {
      v22 = (unsigned __int64 *)v8[8];
      v23 = v8[12] + 8;
      if ( v23 > (unsigned __int64)v22 )
      {
        do
        {
          v24 = *v22++;
          j_j___libc_free_0(v24);
        }
        while ( v23 > (unsigned __int64)v22 );
        v21 = v8[3];
      }
      j_j___libc_free_0(v21);
    }
    v25 = v8[2];
    if ( v25 )
      j_j___libc_free_0_0(v25);
    j_j___libc_free_0((unsigned __int64)v8);
  }
  return v13;
}
