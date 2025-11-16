// Function: sub_7FA4E0
// Address: 0x7fa4e0
//
_QWORD *__fastcall sub_7FA4E0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  const __m128i *v6; // rax
  __m128i *v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __m128i *v10; // r12
  _BYTE *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax

  sub_72BA30(byte_4F06A51[0]);
  v6 = (const __m128i *)sub_72CBE0();
  v7 = sub_73C570(v6, 1);
  v8 = sub_72D2E0(v7);
  v9 = sub_7E1C10();
  v10 = (__m128i *)sub_73E130(a1, v9);
  v11 = sub_73E130(a2, v8);
  v10[1].m128i_i64[0] = (__int64)v11;
  v12 = (__int64)v11;
  *((_QWORD *)v11 + 2) = sub_73A8E0(a3, byte_4F06A51[0]);
  if ( (unsigned int)sub_731770((__int64)v10, 0, v13, v14, v15, v16)
    || (unsigned int)sub_731770(v12, 0, v17, v18, v19, v20) )
  {
    v21 = sub_7E1C10();
    return sub_7F89D0("__gen_nvvm_memcpy", &qword_4F18AD8, v21, v10);
  }
  else
  {
    v23 = *(_DWORD *)(a4 + 136);
    switch ( v23 )
    {
      case 16:
        v25 = sub_7E1C10();
        return sub_7F89D0("__gen_nvvm_memcpy_aligned16", &qword_4F18AD0, v25, v10);
      case 8:
        v27 = sub_7E1C10();
        return sub_7F89D0("__gen_nvvm_memcpy_aligned8", &qword_4F18AC8, v27, v10);
      case 4:
        v28 = sub_7E1C10();
        return sub_7F89D0("__gen_nvvm_memcpy_aligned4", &qword_4F18AC0, v28, v10);
      case 2:
        v26 = sub_7E1C10();
        return sub_7F89D0("__gen_nvvm_memcpy_aligned2", &qword_4F18AB8, v26, v10);
      default:
        v24 = sub_7E1C10();
        return sub_7F89D0("__gen_nvvm_memcpy_aligned1", &qword_4F18AB0, v24, v10);
    }
  }
}
