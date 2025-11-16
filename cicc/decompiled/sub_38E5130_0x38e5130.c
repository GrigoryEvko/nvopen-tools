// Function: sub_38E5130
// Address: 0x38e5130
//
void __fastcall sub_38E5130(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 *v4; // r14
  unsigned __int64 v5; // r12
  int v6; // r13d
  unsigned int v7; // eax
  void (__fastcall *v8)(__int64, _QWORD); // rax
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  int v11; // r13d
  int v12; // eax
  void (__fastcall *v13)(_BYTE *, _QWORD); // rax
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  _QWORD *v17; // [rsp+10h] [rbp-1D0h]
  int v18; // [rsp+1Ch] [rbp-1C4h]
  _BYTE *v19; // [rsp+20h] [rbp-1C0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-1B8h]
  _BYTE v21[16]; // [rsp+30h] [rbp-1B0h] BYREF
  _BYTE v22[16]; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 *v23; // [rsp+50h] [rbp-190h]
  __int64 v24; // [rsp+60h] [rbp-180h] BYREF
  __int64 *v25; // [rsp+80h] [rbp-160h]
  __int64 v26; // [rsp+90h] [rbp-150h] BYREF
  __int64 *v27; // [rsp+A0h] [rbp-140h]
  __int64 v28; // [rsp+B0h] [rbp-130h] BYREF
  unsigned __int64 v29; // [rsp+C0h] [rbp-120h]
  unsigned __int64 v30; // [rsp+D8h] [rbp-108h]
  unsigned int v31; // [rsp+E0h] [rbp-100h]
  char v32; // [rsp+E8h] [rbp-F8h] BYREF

  v3 = sub_16E8CB0();
  v4 = *(__int64 **)a1;
  v5 = *(_QWORD *)(a1 + 8);
  v17 = v3;
  v6 = sub_16CE270(*(__int64 **)a1, v5);
  v18 = sub_16CE270(*(__int64 **)(a2 + 344), *(_QWORD *)(a2 + 584));
  v7 = sub_16CE270(v4, v5);
  if ( !*(_QWORD *)(a2 + 352) && v7 > 1 )
    sub_16CFB30(v4, *(_QWORD *)(*v4 + 24LL * (v7 - 1) + 16), (__int64)v17);
  if ( *(_QWORD *)(a2 + 576) && *(__int64 **)(a2 + 344) == v4 && v6 == v18 )
  {
    v9 = *(_BYTE **)(a2 + 560);
    if ( v9 )
    {
      v10 = (__int64)&v9[*(_QWORD *)(a2 + 568)];
      v19 = v21;
      sub_38E3110((__int64 *)&v19, v9, v10);
    }
    else
    {
      v21[0] = 0;
      v19 = v21;
      v20 = 0;
    }
    v11 = sub_16CFA40(v4, v5, v6);
    v12 = sub_16CFA40(*(__int64 **)(a2 + 344), *(_QWORD *)(a2 + 584), v18);
    sub_16D0AA0(
      (__int64)v22,
      *(_QWORD *)a1,
      *(_QWORD *)(a1 + 8),
      v19,
      v20,
      v11 - v12 + *(_QWORD *)(a2 + 576) - 1,
      *(_DWORD *)(a1 + 52),
      *(_DWORD *)(a1 + 56),
      *(_BYTE **)(a1 + 64),
      *(_QWORD *)(a1 + 72),
      *(_BYTE **)(a1 + 96),
      *(_QWORD *)(a1 + 104),
      *(_QWORD **)(a1 + 128),
      (__int64)(*(_QWORD *)(a1 + 136) - *(_QWORD *)(a1 + 128)) >> 3,
      0,
      0);
    v13 = *(void (__fastcall **)(_BYTE *, _QWORD))(a2 + 352);
    if ( v13 )
      v13(v22, *(_QWORD *)(a2 + 360));
    else
      sub_16CE370((__int64)v22, 0, v17, 1, 1);
    v14 = v30;
    v15 = v30 + 48LL * v31;
    if ( v30 != v15 )
    {
      do
      {
        v15 -= 48LL;
        v16 = *(_QWORD *)(v15 + 16);
        if ( v16 != v15 + 32 )
          j_j___libc_free_0(v16);
      }
      while ( v14 != v15 );
      v15 = v30;
    }
    if ( (char *)v15 != &v32 )
      _libc_free(v15);
    if ( v29 )
      j_j___libc_free_0(v29);
    if ( v27 != &v28 )
      j_j___libc_free_0((unsigned __int64)v27);
    if ( v25 != &v26 )
      j_j___libc_free_0((unsigned __int64)v25);
    if ( v23 != &v24 )
      j_j___libc_free_0((unsigned __int64)v23);
    if ( v19 != v21 )
      j_j___libc_free_0((unsigned __int64)v19);
  }
  else
  {
    v8 = *(void (__fastcall **)(__int64, _QWORD))(a2 + 352);
    if ( v8 )
      v8(a1, *(_QWORD *)(a2 + 360));
    else
      sub_16CE370(a1, 0, v17, 1, 1);
  }
}
