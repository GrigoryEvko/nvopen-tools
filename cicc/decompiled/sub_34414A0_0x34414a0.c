// Function: sub_34414A0
// Address: 0x34414a0
//
__int64 __fastcall sub_34414A0(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  _QWORD *v5; // rbx
  __int64 (__fastcall *v6)(__int64, __int64); // rax
  int v7; // eax
  unsigned int v8; // r12d
  _QWORD *v10; // rax
  __int64 v11; // r15
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // [rsp+8h] [rbp-38h]

  v5 = a4;
  v6 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 2128LL);
  if ( v6 != sub_302E0D0 )
    a2 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v6)(a1, a2, 0);
  v7 = *(_DWORD *)(a2 + 24);
  LOBYTE(a4) = (unsigned int)(v7 - 37) <= 1 || (unsigned int)(v7 - 13) <= 1;
  v8 = (unsigned int)a4;
  if ( (_BYTE)a4 )
  {
    *a3 = *(_QWORD *)(a2 + 96);
    *v5 += *(_QWORD *)(a2 + 104);
    return v8;
  }
  if ( v7 != 56 )
    return v8;
  v10 = *(_QWORD **)(a2 + 40);
  v11 = *v10;
  v18 = v10[5];
  v12 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD *, _QWORD *))(*(_QWORD *)a1 + 2136LL))(a1, *v10, a3, v5);
  if ( (_BYTE)v12 )
  {
    LOBYTE(v12) = *(_DWORD *)(v18 + 24) == 11 || *(_DWORD *)(v18 + 24) == 35;
    if ( !(_BYTE)v12 )
      return v8;
    v13 = *(_QWORD *)(v18 + 96);
    v14 = *(_DWORD *)(v13 + 32);
    v15 = *(__int64 **)(v13 + 24);
    if ( v14 <= 0x40 )
      goto LABEL_10;
    goto LABEL_16;
  }
  v12 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD *))(*(_QWORD *)a1 + 2136LL))(a1, v18, a3, v5);
  if ( (_BYTE)v12 )
  {
    LOBYTE(v12) = *(_DWORD *)(v11 + 24) == 35 || *(_DWORD *)(v11 + 24) == 11;
    if ( (_BYTE)v12 )
    {
      v17 = *(_QWORD *)(v11 + 96);
      v14 = *(_DWORD *)(v17 + 32);
      v15 = *(__int64 **)(v17 + 24);
      if ( v14 <= 0x40 )
      {
LABEL_10:
        v16 = 0;
        if ( v14 )
          v16 = (__int64)((_QWORD)v15 << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
        goto LABEL_12;
      }
LABEL_16:
      v16 = *v15;
LABEL_12:
      *v5 += v16;
      return v12;
    }
  }
  return v8;
}
