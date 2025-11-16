// Function: sub_1A3E810
// Address: 0x1a3e810
//
__int64 __fastcall sub_1A3E810(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  _QWORD *v28; // rbx
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  _BYTE v31[8]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+8h] [rbp-98h]
  unsigned __int64 v33; // [rsp+10h] [rbp-90h]
  unsigned int v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+1Ch] [rbp-84h]
  __int64 v36; // [rsp+40h] [rbp-60h]
  unsigned __int64 v37; // [rsp+48h] [rbp-58h]
  int v38; // [rsp+54h] [rbp-4Ch]
  int v39; // [rsp+58h] [rbp-48h]

  LODWORD(v10) = 0;
  if ( !(unsigned __int8)sub_1636880(a1, a2) )
  {
    v12 = *(__int64 **)(a1 + 8);
    v13 = *v12;
    v14 = v12[1];
    if ( v13 == v14 )
      goto LABEL_31;
    while ( *(_UNKNOWN **)v13 != &unk_4F9D764 )
    {
      v13 += 16;
      if ( v14 == v13 )
        goto LABEL_31;
    }
    v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
            *(_QWORD *)(v13 + 8),
            &unk_4F9D764);
    v16 = sub_14CF090(v15, a2);
    v17 = *(__int64 **)(a1 + 8);
    v18 = v16;
    v19 = *v17;
    v20 = v17[1];
    if ( v19 == v20 )
      goto LABEL_31;
    while ( *(_UNKNOWN **)v19 != &unk_4F96DB4 )
    {
      v19 += 16;
      if ( v20 == v19 )
        goto LABEL_31;
    }
    v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
            *(_QWORD *)(v19 + 8),
            &unk_4F96DB4);
    v22 = *(__int64 **)(a1 + 8);
    v10 = *(_QWORD *)(v21 + 160);
    v23 = *v22;
    v24 = v22[1];
    if ( v23 == v24 )
LABEL_31:
      BUG();
    while ( *(_UNKNOWN **)v23 != &unk_4F9E06C )
    {
      v23 += 16;
      if ( v24 == v23 )
        goto LABEL_31;
    }
    v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
            *(_QWORD *)(v23 + 8),
            &unk_4F9E06C);
    sub_1A3DCD0((__int64)v31, a1 + 160, a2, v25 + 160, v10, v18, a3, a4, a5, a6, v26, v27, a9, a10);
    if ( v39 == v38 )
    {
      if ( v33 == v32 )
        v28 = (_QWORD *)(v33 + 8LL * v35);
      else
        v28 = (_QWORD *)(v33 + 8LL * v34);
      v29 = sub_15CC2D0((__int64)v31, (__int64)&unk_4F9EE48);
      if ( v33 == v32 )
        v30 = (_QWORD *)(v33 + 8LL * v35);
      else
        v30 = (_QWORD *)(v33 + 8LL * v34);
      for ( ; v30 != v29; ++v29 )
      {
        if ( *v29 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      LOBYTE(v10) = v29 == v28;
    }
    else
    {
      LODWORD(v10) = 1;
    }
    if ( v37 != v36 )
      _libc_free(v37);
    if ( v33 != v32 )
      _libc_free(v33);
  }
  return (unsigned int)v10;
}
