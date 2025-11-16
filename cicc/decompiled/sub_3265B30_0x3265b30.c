// Function: sub_3265B30
// Address: 0x3265b30
//
bool __fastcall sub_3265B30(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v4; // rax
  unsigned __int64 *v5; // r8
  __int64 v6; // rdi
  int v7; // eax
  char v8; // r13
  __int64 v9; // rax
  int v10; // eax
  char v11; // r13
  __int64 v12; // rax
  __m128i v13; // [rsp-58h] [rbp-58h]
  __m128i v14; // [rsp-48h] [rbp-48h]
  unsigned __int64 v15; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v16; // [rsp-30h] [rbp-30h]

  if ( *(_DWORD *)a1 != *(_DWORD *)(a2 + 24) )
    return 0;
  v4 = *(_QWORD *)(a1 + 8);
  v14 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  *(_QWORD *)v4 = v14.m128i_i64[0];
  *(_DWORD *)(v4 + 8) = v14.m128i_i32[2];
  v5 = *(unsigned __int64 **)(a1 + 16);
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  if ( v6 )
  {
    v7 = *(_DWORD *)(v6 + 24);
    if ( v7 == 35 || v7 == 11 )
      goto LABEL_22;
  }
  v16 = 1;
  if ( !v5 )
    v5 = &v15;
  v15 = 0;
  v8 = sub_33D1410(v6, v5);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v8 )
    goto LABEL_27;
  v9 = *(_QWORD *)(a1 + 8);
  v13 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  *(_QWORD *)v9 = v13.m128i_i64[0];
  *(_DWORD *)(v9 + 8) = v13.m128i_i32[2];
  v5 = *(unsigned __int64 **)(a1 + 16);
  v6 = **(_QWORD **)(a2 + 40);
  if ( v6 )
  {
    v10 = *(_DWORD *)(v6 + 24);
    if ( v10 == 11 || v10 == 35 )
    {
LABEL_22:
      if ( v5 )
      {
        v12 = *(_QWORD *)(v6 + 96);
        if ( *((_DWORD *)v5 + 2) > 0x40u || *(_DWORD *)(v12 + 32) > 0x40u )
        {
          sub_C43990((__int64)v5, v12 + 24);
        }
        else
        {
          *v5 = *(_QWORD *)(v12 + 24);
          *((_DWORD *)v5 + 2) = *(_DWORD *)(v12 + 32);
        }
      }
      goto LABEL_27;
    }
  }
  v16 = 1;
  if ( !v5 )
    v5 = &v15;
  v15 = 0;
  v11 = sub_33D1410(v6, v5);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  result = 0;
  if ( v11 )
  {
LABEL_27:
    result = 1;
    if ( *(_BYTE *)(a1 + 28) )
      return (*(_DWORD *)(a1 + 24) & *(_DWORD *)(a2 + 28)) == *(_DWORD *)(a1 + 24);
  }
  return result;
}
