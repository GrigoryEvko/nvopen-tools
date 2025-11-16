// Function: sub_20A15A0
// Address: 0x20a15a0
//
__int64 __fastcall sub_20A15A0(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  _QWORD *v5; // rbx
  __int16 v6; // ax
  unsigned int v7; // r12d
  _QWORD *v9; // rax
  __int64 v10; // r15
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // [rsp+8h] [rbp-38h]

  v5 = a4;
  v6 = *(_WORD *)(a2 + 24);
  LOBYTE(a4) = (unsigned __int16)(v6 - 12) <= 1u || (unsigned __int16)(v6 - 34) <= 1u;
  v7 = (unsigned int)a4;
  if ( (_BYTE)a4 )
  {
    *a3 = *(_QWORD *)(a2 + 88);
    *v5 += *(_QWORD *)(a2 + 96);
    return v7;
  }
  if ( v6 != 52 )
    return v7;
  v9 = *(_QWORD **)(a2 + 32);
  v10 = *v9;
  v17 = v9[5];
  v11 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD *, _QWORD *))(*(_QWORD *)a1 + 1104LL))(a1, *v9, a3, v5);
  if ( (_BYTE)v11 )
  {
    LOBYTE(v11) = *(_WORD *)(v17 + 24) == 10 || *(_WORD *)(v17 + 24) == 32;
    if ( !(_BYTE)v11 )
      return v7;
    v12 = *(_QWORD *)(v17 + 88);
    v13 = *(_DWORD *)(v12 + 32);
    v14 = *(__int64 **)(v12 + 24);
    if ( v13 > 0x40 )
      goto LABEL_8;
    goto LABEL_13;
  }
  v11 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD *))(*(_QWORD *)a1 + 1104LL))(a1, v17, a3, v5);
  if ( (_BYTE)v11 )
  {
    LOBYTE(v11) = *(_WORD *)(v10 + 24) == 10 || *(_WORD *)(v10 + 24) == 32;
    if ( (_BYTE)v11 )
    {
      v16 = *(_QWORD *)(v10 + 88);
      v13 = *(_DWORD *)(v16 + 32);
      v14 = *(__int64 **)(v16 + 24);
      if ( v13 > 0x40 )
      {
LABEL_8:
        v15 = *v14;
LABEL_9:
        *v5 += v15;
        return v11;
      }
LABEL_13:
      v15 = (__int64)((_QWORD)v14 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13);
      goto LABEL_9;
    }
  }
  return v7;
}
