// Function: sub_39A6B70
// Address: 0x39a6b70
//
__int64 __fastcall sub_39A6B70(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // r14
  unsigned __int8 v12; // dl
  __int64 v13; // rax
  void *v14; // rcx
  size_t v15; // rdx
  size_t v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // r15
  __int64 v20; // rax
  void *v21; // rax
  size_t v22; // rdx

  v5 = sub_39A5A90((__int64)a1, *(_WORD *)(a3 + 2), a2, 0);
  v6 = v5;
  if ( *(_WORD *)(a3 + 2) == 48 )
    sub_39A6760(a1, v5, *(_QWORD *)(a3 + 8 * (1LL - *(unsigned int *)(a3 + 8))), 73);
  result = *(unsigned int *)(a3 + 8);
  v8 = *(_QWORD *)(a3 - 8 * result);
  if ( v8 )
  {
    sub_161E970(v8);
    result = *(unsigned int *)(a3 + 8);
    if ( v9 )
    {
      v13 = -result;
      v14 = *(void **)(a3 + 8 * v13);
      if ( v14 )
      {
        v14 = (void *)sub_161E970(*(_QWORD *)(a3 + 8 * v13));
        v16 = v15;
      }
      else
      {
        v16 = 0;
      }
      sub_39A3F30(a1, v6, 3, v14, v16);
      result = *(unsigned int *)(a3 + 8);
    }
  }
  v10 = *(_QWORD *)(a3 + 8 * (2 - result));
  if ( v10 )
  {
    if ( *(_BYTE *)v10 != 1 )
      goto LABEL_15;
    v11 = *(_QWORD *)(v10 + 136);
    v12 = *(_BYTE *)(v11 + 16);
    if ( v12 == 13 )
      return sub_39A5150((__int64)a1, v6, v11, *(_QWORD *)(a3 + 8 * (1 - result)));
    if ( v12 <= 3u )
    {
      result = *(_BYTE *)(v11 + 33) & 3;
      if ( (_BYTE)result != 1 )
      {
        v17 = sub_145CBF0(a1 + 11, 16, 16);
        v18 = a1[24];
        *(_QWORD *)v17 = 0;
        v19 = (__int64 *)v17;
        *(_DWORD *)(v17 + 8) = 0;
        v20 = sub_396EAF0(v18, v11);
        sub_39A39D0((__int64)a1, v19, v20);
        sub_39A35E0((__int64)a1, v19, 11, 159);
        return sub_39A4520(a1, v6, 2, (__int64 **)v19);
      }
    }
    else
    {
LABEL_15:
      result = *(unsigned __int16 *)(a3 + 2);
      if ( (_DWORD)result == 16646 )
      {
        v21 = (void *)sub_161E970(v10);
        return sub_39A3F30(a1, v6, 8464, v21, v22);
      }
      else if ( (_DWORD)result == 16647 )
      {
        return sub_39A6D90(a1, v6, v10);
      }
    }
  }
  return result;
}
