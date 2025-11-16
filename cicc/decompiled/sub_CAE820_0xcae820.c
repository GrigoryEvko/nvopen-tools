// Function: sub_CAE820
// Address: 0xcae820
//
__int64 __fastcall sub_CAE820(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbp
  __int64 result; // rax
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  _QWORD *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdx
  _QWORD *v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // [rsp-60h] [rbp-60h]
  __int64 v22; // [rsp-58h] [rbp-58h] BYREF
  _QWORD *v23; // [rsp-40h] [rbp-40h]
  _QWORD v24[6]; // [rsp-30h] [rbp-30h] BYREF

  result = *(_QWORD *)(a1 + 72);
  if ( !result )
  {
    v24[5] = v5;
    v8 = *(_DWORD *)sub_CAD7B0(a1, a2, a3, a4, a5);
    if ( (v8 & 0xFFFFFFF7) == 0 || v8 == 17 )
      goto LABEL_8;
    if ( v8 == 16 )
    {
      a2 = a1;
      sub_CAD6B0((__int64)&v22, a1, v9, v10, v11);
      if ( v23 != v24 )
      {
        a2 = v24[0] + 1LL;
        j_j___libc_free_0(v23, v24[0] + 1LL);
      }
    }
    v12 = *(_DWORD *)sub_CAD7B0(a1, a2, v9, v10, v11);
    if ( v12 == 8 || v12 == 17 )
    {
LABEL_8:
      v16 = (_QWORD *)sub_CA8A30(a1);
      v18 = *v16;
      v16[10] += 72LL;
      v19 = v16;
      v20 = (v18 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( v19[1] >= (unsigned __int64)(v20 + 72) && v18 )
        *v19 = v20 + 72;
      else
        v20 = sub_9D1E70((__int64)v19, 72, 72, 4);
      v21 = (_QWORD *)v20;
      sub_CAD7C0(v20, 0, *(_QWORD *)(a1 + 8), 0, 0, v17, 0);
      *v21 = &unk_49DCC78;
      *(_QWORD *)(a1 + 72) = v21;
      return (__int64)v21;
    }
    else
    {
      result = sub_CAE810(a1, a2, v13, v14, v15);
      *(_QWORD *)(a1 + 72) = result;
    }
  }
  return result;
}
