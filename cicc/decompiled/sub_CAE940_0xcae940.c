// Function: sub_CAE940
// Address: 0xcae940
//
_QWORD *__fastcall sub_CAE940(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rax
  _QWORD *v20; // r12
  unsigned int *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  unsigned __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  _QWORD *v35; // rax
  __int64 v36; // rcx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  _QWORD v39[3]; // [rsp+0h] [rbp-50h] BYREF
  __int64 *v40; // [rsp+18h] [rbp-38h]
  char v41; // [rsp+20h] [rbp-30h]
  char v42; // [rsp+21h] [rbp-2Fh]
  __int64 v43; // [rsp+28h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a1 + 80);
  if ( v5 )
    return (_QWORD *)v5;
  v8 = sub_CAE820(a1, a2, a3, a4, a5);
  if ( !v8 )
  {
    v25 = sub_CAD7B0(a1, a2, v9, v10, v11);
    v42 = 1;
    v41 = 3;
    v39[0] = "Null key in Key Value.";
    sub_CA8D00(a1, (__int64)v39, v25, v26, v27);
    goto LABEL_5;
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  if ( !(unsigned __int8)sub_CA94D0(a1) )
  {
    v21 = (unsigned int *)sub_CAD7B0(a1, a2, v12, v13, v14);
    v24 = *v21;
    if ( (unsigned int)v24 <= 0x10 )
    {
      v22 = 100609;
      if ( _bittest64(&v22, v24) )
        goto LABEL_5;
    }
    else if ( (_DWORD)v24 == 17 )
    {
      v28 = a1;
      sub_CAD6B0((__int64)v39, a1, v24, v22, v23);
      if ( v40 != &v43 )
      {
        v28 = v43 + 1;
        j_j___libc_free_0(v40, v43 + 1);
      }
      if ( ((*(_DWORD *)sub_CAD7B0(a1, v28, v29, v30, v31) - 8) & 0xFFFFFFF7) != 0 )
      {
        v38 = sub_CAE810(a1, v28, v32, v33, v34);
        *(_QWORD *)(a1 + 80) = v38;
        return (_QWORD *)v38;
      }
      v35 = (_QWORD *)sub_CA8A30(a1);
      v36 = *v35;
      v35[10] += 72LL;
      v18 = v35;
      v37 = (v36 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( v18[1] >= v37 + 72 && v36 )
      {
        *v18 = v37 + 72;
        v20 = (_QWORD *)((v36 + 15) & 0xFFFFFFFFFFFFFFF0LL);
        goto LABEL_8;
      }
      goto LABEL_14;
    }
    v42 = 1;
    v39[0] = "Unexpected token in Key Value.";
    v41 = 3;
    sub_CA8D00(a1, (__int64)v39, (__int64)v21, v22, v23);
  }
LABEL_5:
  v15 = (_QWORD *)sub_CA8A30(a1);
  v17 = *v15;
  v15[10] += 72LL;
  v18 = v15;
  v19 = (v17 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( v18[1] < v19 + 72 || !v17 )
  {
LABEL_14:
    v20 = (_QWORD *)sub_9D1E70((__int64)v18, 72, 72, 4);
    goto LABEL_8;
  }
  *v18 = v19 + 72;
  v20 = (_QWORD *)((v17 + 15) & 0xFFFFFFFFFFFFFFF0LL);
LABEL_8:
  sub_CAD7C0((__int64)v20, 0, *(_QWORD *)(a1 + 8), 0, 0, v16, 0);
  *v20 = &unk_49DCC78;
  *(_QWORD *)(a1 + 80) = v20;
  return v20;
}
