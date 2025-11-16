// Function: sub_19DEE70
// Address: 0x19dee70
//
__int64 __fastcall sub_19DEE70(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __m128i a5, __m128i a6, double a7)
{
  __int64 v9; // rbx
  unsigned __int8 v10; // al
  __int64 result; // rax
  __int64 **v12; // rbx
  __int64 *v13; // rcx
  __int64 *v14; // rbx
  __int64 v15; // [rsp+8h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 24 * (a3 + 1 - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v10 = *(_BYTE *)(v9 + 16);
  if ( v10 <= 0x17u )
    goto LABEL_11;
  if ( v10 == 62 )
    goto LABEL_8;
  if ( v10 != 61 )
    goto LABEL_4;
  if ( (unsigned __int8)sub_14C2730(*(__int64 **)(v9 - 24), a1[1], 0, *a1, a2, a1[2]) )
LABEL_8:
    v9 = *(_QWORD *)(v9 - 24);
  v10 = *(_BYTE *)(v9 + 16);
  if ( v10 <= 0x17u )
  {
LABEL_11:
    if ( v10 != 5 || *(_WORD *)(v9 + 18) != 11 )
      return 0;
    goto LABEL_12;
  }
LABEL_4:
  if ( v10 != 35 )
    return 0;
LABEL_12:
  if ( sub_19DD640((__int64)a1, v9, (__int64 *)a2) && (unsigned int)sub_14C3880(v9, a1[1], *a1, a2, a1[2]) != 2 )
    return 0;
  if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
    v12 = *(__int64 ***)(v9 - 8);
  else
    v12 = (__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
  v13 = *v12;
  v14 = v12[3];
  v15 = (__int64)v13;
  result = sub_19DDCB0(a1, a2, a3, v13, (__int64)v14, a4, a5, a6, a7);
  if ( !result )
  {
    if ( v14 != (__int64 *)v15 )
      return sub_19DDCB0(a1, a2, a3, v14, v15, a4, a5, a6, a7);
    return 0;
  }
  return result;
}
