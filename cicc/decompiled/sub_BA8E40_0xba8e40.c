// Function: sub_BA8E40
// Address: 0xba8e40
//
__int64 __fastcall sub_BA8E40(__int64 a1, _QWORD *a2, size_t a3)
{
  unsigned int v4; // eax
  unsigned int v5; // edx
  _QWORD *v6; // rcx
  __int64 v7; // rbx
  __int64 result; // rax
  __int64 v9; // rax
  unsigned int v10; // r8d
  _QWORD *v11; // rcx
  _QWORD *v12; // rbx
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 *v16; // rax
  __int64 v17; // rcx
  int v18; // edx
  _QWORD *v19; // [rsp+0h] [rbp-70h]
  unsigned int v20; // [rsp+Ch] [rbp-64h]
  _QWORD v21[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v4 = sub_C92610(a2, a3);
  v5 = sub_C92740(a1 + 288, a2, a3, v4);
  v6 = (_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL * v5);
  v7 = *v6;
  if ( !*v6 )
  {
LABEL_6:
    v19 = v6;
    v20 = v5;
    v9 = sub_C7D670(a3 + 17, 8);
    v10 = v20;
    v11 = v19;
    v12 = (_QWORD *)v9;
    if ( a3 )
    {
      memcpy((void *)(v9 + 16), a2, a3);
      v10 = v20;
      v11 = v19;
    }
    *((_BYTE *)v12 + a3 + 16) = 0;
    *v12 = a3;
    v12[1] = 0;
    *v11 = v12;
    ++*(_DWORD *)(a1 + 300);
    v13 = (__int64 *)(*(_QWORD *)(a1 + 288) + 8LL * (unsigned int)sub_C929D0(a1 + 288, v10));
    v7 = *v13;
    if ( *v13 != -8 )
      goto LABEL_10;
    do
    {
      do
      {
        v7 = v13[1];
        ++v13;
      }
      while ( v7 == -8 );
LABEL_10:
      ;
    }
    while ( !v7 );
    result = *(_QWORD *)(v7 + 8);
    if ( result )
      return result;
    goto LABEL_12;
  }
  if ( v7 == -8 )
  {
    --*(_DWORD *)(a1 + 304);
    goto LABEL_6;
  }
  result = *(_QWORD *)(v7 + 8);
  if ( result )
    return result;
LABEL_12:
  v21[0] = a2;
  v22 = 261;
  v21[1] = a3;
  v14 = sub_22077B0(64);
  v15 = v14;
  if ( v14 )
    sub_B919A0(v14, (__int64)v21);
  *(_QWORD *)(v7 + 8) = v15;
  *(_QWORD *)(v15 + 48) = a1;
  v16 = *(__int64 **)(v7 + 8);
  v17 = *(_QWORD *)(a1 + 72);
  v16[1] = a1 + 72;
  v17 &= 0xFFFFFFFFFFFFFFF8LL;
  *v16 = v17 | *v16 & 7;
  *(_QWORD *)(v17 + 8) = v16;
  *(_QWORD *)(a1 + 72) = *(_QWORD *)(a1 + 72) & 7LL | (unsigned __int64)v16;
  if ( a3 != 17 )
    return *(_QWORD *)(v7 + 8);
  if ( *a2 ^ 0x646F6D2E6D766C6CLL | a2[1] ^ 0x67616C662E656C75LL || (v18 = 0, *((_BYTE *)a2 + 16) != 115) )
    v18 = 1;
  result = *(_QWORD *)(v7 + 8);
  if ( !v18 )
  {
    *(_QWORD *)(a1 + 864) = result;
    return *(_QWORD *)(v7 + 8);
  }
  return result;
}
