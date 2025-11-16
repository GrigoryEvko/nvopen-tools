// Function: sub_386AB30
// Address: 0x386ab30
//
__int64 __fastcall sub_386AB30(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // r13
  unsigned int v7; // r14d
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rcx
  int v14; // esi
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdi
  int v18; // eax
  int v19; // r8d
  __int64 v20[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a2 - 24);
  if ( !v4 )
    BUG();
  if ( *(_BYTE *)(v4 + 16) > 0x10u )
  {
    v11 = *(_QWORD *)(a1 + 40);
    v12 = *(_DWORD *)(v11 + 24);
    if ( !v12 )
      return sub_386A280(a1, (__int64 *)a2, a3, a4);
    v13 = *(_QWORD *)(v11 + 8);
    v14 = v12 - 1;
    v15 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( v4 != *v16 )
    {
      v18 = 1;
      while ( v17 != -8 )
      {
        v19 = v18 + 1;
        v15 = v14 & (v18 + v15);
        v16 = (__int64 *)(v13 + 16LL * v15);
        v17 = *v16;
        if ( v4 == *v16 )
          goto LABEL_9;
        v18 = v19;
      }
      return sub_386A280(a1, (__int64 *)a2, a3, a4);
    }
LABEL_9:
    v4 = v16[1];
    if ( !v4 )
      return sub_386A280(a1, (__int64 *)a2, a3, a4);
  }
  v7 = sub_15FC090((unsigned int)*(unsigned __int8 *)(a2 + 16) - 24, (_QWORD *)v4, *(_QWORD *)a2);
  if ( !(_BYTE)v7 )
    return sub_386A280(a1, (__int64 *)a2, a3, a4);
  v9 = sub_15A46C0((unsigned int)*(unsigned __int8 *)(a2 + 16) - 24, (__int64 ***)v4, *(__int64 ***)a2, 0);
  if ( !v9 )
    return sub_386A280(a1, (__int64 *)a2, a3, a4);
  v10 = *(_QWORD *)(a1 + 40);
  v20[0] = a2;
  sub_38526A0(v10, v20)[1] = v9;
  return v7;
}
