// Function: sub_EA3870
// Address: 0xea3870
//
_QWORD *__fastcall sub_EA3870(__int64 a1, __int64 a2)
{
  void *v3; // rax
  __int64 *v4; // r13
  unsigned __int64 v5; // r14
  int v6; // r15d
  unsigned int v7; // eax
  __int64 (__fastcall *v8)(__int64, _QWORD); // rax
  _QWORD *result; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  int v12; // r15d
  int v13; // eax
  void (__fastcall *v14)(__int64, _BYTE *); // rax
  _BYTE *v15; // rsi
  __int64 v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-1D0h]
  int v20; // [rsp+Ch] [rbp-1C4h]
  _BYTE *v21[2]; // [rsp+10h] [rbp-1C0h] BYREF
  _QWORD v22[2]; // [rsp+20h] [rbp-1B0h] BYREF
  _BYTE v23[16]; // [rsp+30h] [rbp-1A0h] BYREF
  _QWORD *v24; // [rsp+40h] [rbp-190h]
  _QWORD v25[4]; // [rsp+50h] [rbp-180h] BYREF
  __int64 *v26; // [rsp+70h] [rbp-160h]
  __int64 v27; // [rsp+80h] [rbp-150h] BYREF
  __int64 *v28; // [rsp+90h] [rbp-140h]
  __int64 v29; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v30; // [rsp+B0h] [rbp-120h]
  __int64 v31; // [rsp+C0h] [rbp-110h]
  __int64 v32; // [rsp+C8h] [rbp-108h]
  unsigned int v33; // [rsp+D0h] [rbp-100h]
  char v34; // [rsp+D8h] [rbp-F8h] BYREF

  v3 = sub_CB72A0();
  v4 = *(__int64 **)a1;
  v5 = *(_QWORD *)(a1 + 8);
  v19 = (__int64)v3;
  v6 = sub_C8ED90(*(__int64 **)a1, v5);
  v20 = sub_C8ED90(*(__int64 **)(a2 + 248), *(_QWORD *)(a2 + 504));
  v7 = sub_C8ED90(v4, v5);
  if ( !*(_QWORD *)(a2 + 256) && v7 > 1 )
    sub_C904A0(v4, *(_QWORD *)(*v4 + 24LL * (v7 - 1) + 16), v19);
  if ( *(_QWORD *)(a2 + 496) && v6 == v20 )
  {
    v10 = *(_BYTE **)(a2 + 480);
    v11 = (__int64)&v10[*(_QWORD *)(a2 + 488)];
    v21[0] = v22;
    sub_EA2A30((__int64 *)v21, v10, v11);
    v12 = sub_C90410(v4, v5, v6);
    v13 = sub_C90410(*(__int64 **)(a2 + 248), *(_QWORD *)(a2 + 504), v20);
    sub_C91410(
      (__int64)v23,
      *(_QWORD *)a1,
      *(_QWORD *)(a1 + 8),
      v21[0],
      (__int64)v21[1],
      v12 - v13 + *(_QWORD *)(a2 + 496) - 1,
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
    v14 = *(void (__fastcall **)(__int64, _BYTE *))(a2 + 256);
    if ( v14 )
    {
      v15 = *(_BYTE **)(a2 + 264);
      v14(a1, v15);
    }
    else
    {
      v15 = v23;
      sub_E66480(*(_QWORD *)(a2 + 224), (__int64)v23);
    }
    v16 = v32;
    v17 = v32 + 48LL * v33;
    if ( v32 != v17 )
    {
      do
      {
        v17 -= 48;
        v18 = *(_QWORD *)(v17 + 16);
        if ( v18 != v17 + 32 )
        {
          v15 = (_BYTE *)(*(_QWORD *)(v17 + 32) + 1LL);
          j_j___libc_free_0(v18, v15);
        }
      }
      while ( v16 != v17 );
      v17 = v32;
    }
    if ( (char *)v17 != &v34 )
      _libc_free(v17, v15);
    if ( v30 )
      j_j___libc_free_0(v30, v31 - v30);
    if ( v28 != &v29 )
      j_j___libc_free_0(v28, v29 + 1);
    if ( v26 != &v27 )
      j_j___libc_free_0(v26, v27 + 1);
    result = v25;
    if ( v24 != v25 )
      result = (_QWORD *)j_j___libc_free_0(v24, v25[0] + 1LL);
    if ( (_QWORD *)v21[0] != v22 )
      return (_QWORD *)j_j___libc_free_0(v21[0], v22[0] + 1LL);
  }
  else
  {
    v8 = *(__int64 (__fastcall **)(__int64, _QWORD))(a2 + 256);
    if ( v8 )
      return (_QWORD *)v8(a1, *(_QWORD *)(a2 + 264));
    else
      return (_QWORD *)sub_E66480(*(_QWORD *)(a2 + 224), a1);
  }
  return result;
}
