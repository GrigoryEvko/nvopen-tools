// Function: sub_8B4E50
// Address: 0x8b4e50
//
__int64 __fastcall sub_8B4E50(unsigned __int64 a1, __int64 a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // r8
  int v14; // eax
  char v15; // al
  __m128i *v16; // rax
  __int64 v17; // rbx
  _BYTE *v18; // rax
  _BYTE *i; // r12
  __int64 *v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v8 = sub_730E00(a2);
  if ( *(_BYTE *)(v8 + 173) != 12 )
    return 1;
  v10 = v8;
  if ( *(_BYTE *)(v8 + 176) )
    return 1;
  v11 = sub_8A4360(a4, a3, (unsigned int *)(v8 + 184), 0, 0);
  v12 = v11;
  if ( (v11[3] & 1) != 0 )
    return v12[4] == a1;
  v13 = v11[4];
  v20 = v11;
  if ( !v13 )
  {
    v14 = sub_8D2780(*(_QWORD *)(v10 + 128));
    v12 = v20;
    if ( !v14 )
    {
      v15 = *((_BYTE *)v20 + 24);
      if ( (v15 & 1) == 0 )
      {
        v20[4] = a1;
        *((_BYTE *)v20 + 24) = v15 | 1;
        if ( dword_4D04804 )
        {
          v16 = (__m128i *)sub_72BA30(byte_4F06A51[0]);
          sub_8B3500(v16, *(_QWORD *)(v10 + 128), a3, a4, a5 | 0x100);
          return 1;
        }
        return 1;
      }
      return v12[4] == a1;
    }
    v17 = *(_QWORD *)(v10 + 128);
    v18 = sub_724D80(1);
    for ( i = v18; *(_BYTE *)(v17 + 140) == 12; v17 = *(_QWORD *)(v17 + 160) )
      ;
    sub_72BBE0((__int64)v18, a1, *(_BYTE *)(v17 + 160));
    *((_BYTE *)v20 + 24) &= ~1u;
    v20[4] = (__int64)i;
    return 1;
  }
  v21 = v11[4];
  result = sub_8D2780(*(_QWORD *)(v13 + 128));
  if ( (_DWORD)result )
    return (unsigned int)sub_621100(v21, a1) == 0;
  return result;
}
