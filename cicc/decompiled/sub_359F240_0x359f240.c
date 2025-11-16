// Function: sub_359F240
// Address: 0x359f240
//
__int64 __fastcall sub_359F240(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, _QWORD *a5, __int64 a6)
{
  unsigned int v7; // r12d
  __int64 v8; // rdi
  _QWORD *v9; // r15
  unsigned int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64, __int64, _QWORD *, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD); // rax
  int v16; // ebx
  _QWORD *v17; // r15
  unsigned __int64 v18; // rax
  __int64 result; // rax
  __int64 v20; // rdi
  __int64 (__fastcall *v21)(__int64, __int64, __int64, _QWORD *, _BYTE *, _QWORD, __int64 *, _QWORD); // rax
  __int64 v22; // rdi
  __int64 (__fastcall *v23)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *); // rax
  __int64 v24; // r13
  _QWORD *v25; // r12
  _QWORD *v26; // r14
  unsigned __int64 *v27; // rdi
  unsigned __int64 v28; // rdx
  _QWORD *v29; // r12
  _QWORD *v30; // r14
  unsigned __int64 *v31; // rdi
  unsigned __int64 v32; // rdx
  __int64 v33; // rdi
  void (*v34)(); // rax
  __int64 *v35; // [rsp+0h] [rbp-190h]
  unsigned int v37; // [rsp+10h] [rbp-180h]
  __int64 *v38; // [rsp+10h] [rbp-180h]
  int v39; // [rsp+18h] [rbp-178h]
  unsigned int v40; // [rsp+1Ch] [rbp-174h]
  __int64 v44; // [rsp+50h] [rbp-140h]
  unsigned int v45; // [rsp+58h] [rbp-138h]
  unsigned int v46; // [rsp+5Ch] [rbp-134h]
  __int64 v47; // [rsp+60h] [rbp-130h]
  _QWORD *v48; // [rsp+68h] [rbp-128h]
  __int64 v49[3]; // [rsp+78h] [rbp-118h] BYREF
  char v50; // [rsp+90h] [rbp-100h] BYREF
  _BYTE *v51; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v52; // [rsp+B8h] [rbp-D8h]
  _BYTE v53[208]; // [rsp+C0h] [rbp-D0h] BYREF

  v49[1] = (__int64)&v50;
  v49[2] = 0x400000000LL;
  v39 = *(_DWORD *)(a3 + 8);
  v45 = v39 - 1;
  v7 = v39 - 1;
  v44 = (__int64)a4;
  v47 = (__int64)a4;
  v46 = 0;
  do
  {
    v8 = a1[9];
    v9 = (_QWORD *)v47;
    v48 = (_QWORD *)v44;
    v47 = *(_QWORD *)(*(_QWORD *)a3 + 8LL * v7);
    v44 = *(_QWORD *)(*a5 + 8LL * v46);
    v51 = v53;
    v52 = 0x400000000LL;
    v10 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _BYTE **))(*(_QWORD *)v8 + 32LL))(v8, v7 + 1, v47, &v51);
    v13 = v10;
    LOWORD(v13) = BYTE1(v10);
    if ( BYTE1(v10) )
    {
      if ( (_BYTE)v10 )
      {
        v14 = a1[4];
        v15 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v14 + 368LL);
        v49[0] = 0;
        v16 = v15(v14, v47, v9, 0, v51, (unsigned int)v52, v49, 0);
        if ( v49[0] )
          sub_B91220((__int64)v49, v49[0]);
        sub_359BB30(v44, v47);
      }
      else
      {
        sub_2E33F80(v47, v44, -1, v13, v11, v12);
        sub_2E33650(v47, (__int64)v9);
        sub_2E33650((__int64)v48, v44);
        v22 = a1[4];
        v23 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)v22 + 368LL);
        v49[0] = 0;
        v16 = v23(v22, v47, v44, 0, v51, (unsigned int)v52, v49);
        if ( v49[0] )
          sub_B91220((__int64)v49, v49[0]);
        sub_359BB30(v44, (__int64)v48);
        if ( v9 != v48 )
        {
          v24 = (__int64)(v48 + 5);
          if ( v48 + 6 != (_QWORD *)v48[7] )
          {
            v35 = a1;
            v37 = v7;
            v25 = (_QWORD *)v48[7];
            do
            {
              v26 = v25;
              v25 = (_QWORD *)v25[1];
              sub_2E31080(v24, (__int64)v26);
              v27 = (unsigned __int64 *)v26[1];
              v28 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
              *v27 = v28 | *v27 & 7;
              *(_QWORD *)(v28 + 8) = v27;
              *v26 &= 7uLL;
              v26[1] = 0;
              sub_2E310F0(v24);
            }
            while ( v48 + 6 != v25 );
            v7 = v37;
            a1 = v35;
          }
          sub_2E32710(v48);
        }
        if ( a4 == v9 )
        {
          v33 = a1[9];
          v34 = *(void (**)())(*(_QWORD *)v33 + 64LL);
          if ( v34 != nullsub_1682 )
            ((void (__fastcall *)(__int64, __int64))v34)(v33, a1[5]);
          a1[8] = 0;
        }
        if ( v9 + 6 != (_QWORD *)v9[7] )
        {
          v38 = a1;
          v40 = v7;
          v29 = (_QWORD *)v9[7];
          do
          {
            v30 = v29;
            v29 = (_QWORD *)v29[1];
            sub_2E31080((__int64)(v9 + 5), (__int64)v30);
            v31 = (unsigned __int64 *)v30[1];
            v32 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
            *v31 = v32 | *v31 & 7;
            *(_QWORD *)(v32 + 8) = v31;
            *v30 &= 7uLL;
            v30[1] = 0;
            sub_2E310F0((__int64)(v9 + 5));
          }
          while ( v9 + 6 != v29 );
          v7 = v40;
          a1 = v38;
        }
        sub_2E32710(v9);
      }
    }
    else
    {
      sub_2E33F80(v47, v44, -1, v13, v11, v12);
      v20 = a1[4];
      v21 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD *, _BYTE *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v20 + 368LL);
      v49[0] = 0;
      v16 = v21(v20, v47, v44, v9, v51, (unsigned int)v52, v49, 0);
      if ( v49[0] )
        sub_B91220((__int64)v49, v49[0]);
    }
    v17 = (_QWORD *)(*(_QWORD *)(v47 + 48) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (_QWORD *)(v47 + 48) != v17 && v16 )
    {
      do
      {
        sub_359F080(a1, (__int64)v17, 0, v7, 0, a6);
        v18 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
        v17 = (_QWORD *)v18;
        --v16;
      }
      while ( v16 && v47 + 48 != v18 );
    }
    if ( v51 != v53 )
      _libc_free((unsigned __int64)v51);
    ++v46;
    --v7;
    result = v46;
  }
  while ( v45 >= v46 );
  if ( a1[8] )
  {
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1[9] + 56LL))(a1[9], *(_QWORD *)(*(_QWORD *)a3 + 8LL * v45));
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1[9] + 48LL))(a1[9], (unsigned int)-v39);
  }
  return result;
}
